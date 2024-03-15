import clip.clip as clip
import torch
from tqdm import tqdm

from .. import datasets, templates, utils


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        self.model, self.train_preprocess, self.val_preprocess = clip.load(
            args.model, args.device, jit=False
        )

        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def save(self, filename):
        print(f"Saving image encoder to {filename}")
        utils.torch_save(self, filename)

    def load(self, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(self, filename)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f"Saving classification head to {filename}")
        utils.torch_save(self, filename)

    def load(self, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(self, filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            features = self.image_encoder(inputs)
            outputs = self.classification_head(features)
            return outputs, features
        else:
            outputs = self.classification_head(inputs)
            return outputs

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    def load(self, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(self, filename)
    

def create_clip_head_weight(model, dataset, templates=None):
    logit_scale = model.logit_scale
    dat_names = dataset.classnames
    if templates is not None:
        template = templates
    else:
        template = dataset.templates
    
    zeroshot_weights = []
    for classname in dat_names:
        texts = []
        for t in template:
            texts.append(t(classname))
        texts = clip.tokenize(texts).cuda()  # tokenize
        embeddings = model.encode_text(texts)  # embed with text encoder
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.mean(dim=0, keepdim=True)
        # embeddings = embeddings / embeddings.norm()

        zeroshot_weights.append(embeddings)

    zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
    zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
    zeroshot_weights = zeroshot_weights * logit_scale.exp()

    zeroshot_weights = zeroshot_weights.squeeze().float()
    zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)
    return zeroshot_weights, classification_head


def classify_head(weights, inputs, normalize=True):
    if normalize:
        inputs = inputs / inputs.norm(dim=-1, keepdim=True)
    return inputs @ weights.t()


def create_zeroshot_classifier_head(args, clip_model=None, dataset=None):
    if clip_model is None:
        clip_model = ImageEncoder(args, keep_lang=True).model
    logit_scale = clip_model.logit_scale
    # datasets
    dataset_name = args.train_dataset if dataset is None else dataset
    dataset_class = getattr(datasets, dataset_name)
    dataset = dataset_class(
        None,
        location=args.data_location,
        batch_size=args.batch_size,
        # classnames=args.classnames,
    )
    dat_names = dataset.classnames
    
    # templates
    if args.template is not None:
        template = getattr(templates, args.template)
    else:
        template = dataset.templates[:1]

    device = args.device
    clip_model.eval()
    clip_model.to(device)

    print(f"Getting zeroshot weights for {dataset_name}.")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dat_names):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = clip.tokenize(texts).to(device)  # tokenize
            embeddings = clip_model.encode_text(texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def create_image_classifier(args, initialize=True, setnone=False, buffer=False):
    image_encoder = ImageEncoder(args, keep_lang=True)
    if setnone:
        classification_head = None
    elif initialize:
        classification_head = create_zeroshot_classifier_head(args, image_encoder.model)
    elif buffer:
        zeroshot_weights = torch.zeros(buffer, 512)
        classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)
    else:
        zeroshot_weights = torch.zeros(args.n_class, 512)
        classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)
    delattr(image_encoder.model, "transformer")
    classifier = ImageClassifier(
        image_encoder, classification_head, process_images=True
    )
    return classifier

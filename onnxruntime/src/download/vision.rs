//! Module defining computer vision models available to download.

use super::ModelUrl;

/// Computer vision model
#[derive(Debug, Clone)]
pub enum Vision {
    /// Image classification model
    ImageClassification(ImageClassificationModel),
}
/// Image classification model
///
/// > This collection of models take images as input, then classifies the major objects in the images
/// > into 1000 object categories such as keyboard, mouse, pencil, and many animals.
///
/// Source: [https://github.com/onnx/models#image-classification-](https://github.com/onnx/models#image-classification-)
#[derive(Debug, Clone)]
pub enum ImageClassificationModel {
    /// Handwritten digits prediction using CNN
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/mnist](https://github.com/onnx/models/tree/master/vision/classification/mnist)
    ///
    /// Variant downloaded: ONNX Version 1.3 with Opset Version 8.
    Mnist,
    /// Image classification aimed for mobile targets.
    ///
    /// > MobileNet models perform image classification - they take images as input and classify the major
    /// > object in the image into a set of pre-defined classes. They are trained on ImageNet dataset which
    /// > contains images from 1000 classes. MobileNet models are also very efficient in terms of speed and
    /// > size and hence are ideal for embedded and mobile applications.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/mobilenet](https://github.com/onnx/models/tree/master/vision/classification/mobilenet)
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    MobileNet,
    /// Image classification, trained on ImageNet with 1000 classes.
    ///
    /// > ResNet models provide very high accuracies with affordable model sizes. They are ideal for cases when
    /// > high accuracy of classification is required.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/resnet](https://github.com/onnx/models/tree/master/vision/classification/resnet)
    ResNet(ResNet),
    /// A small CNN with AlexNet level accuracy on ImageNet with 50x fewer parameters.
    ///
    /// > SqueezeNet is a small CNN which achieves AlexNet level accuracy on ImageNet with 50x fewer parameters.
    /// > SqueezeNet requires less communication across servers during distributed training, less bandwidth to
    /// > export a new model from the cloud to an autonomous car and more feasible to deploy on FPGAs and other
    /// > hardware with limited memory.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/squeezenet](https://github.com/onnx/models/tree/master/vision/classification/squeezenet)
    ///
    /// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
    SqueezeNet,
    /// Image classification, trained on ImageNet with 1000 classes.
    ///
    /// > VGG models provide very high accuracies but at the cost of increased model sizes. They are ideal for
    /// > cases when high accuracy of classification is essential and there are limited constraints on model sizes.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/vgg](https://github.com/onnx/models/tree/master/vision/classification/vgg)
    Vgg(Vgg),
    /// Convolutional neural network for classification, which competed in the ImageNet Large Scale Visual Recognition Challenge in 2012.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/alexnet](https://github.com/onnx/models/tree/master/vision/classification/alexnet)
    ///
    /// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
    AlexNet,
    /// Convolutional neural network for classification, which competed in the ImageNet Large Scale Visual Recognition Challenge in 2014.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/googlenet](https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/googlenet)
    ///
    /// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
    GoogleNet,
    /// Variant of AlexNet, it's the name of a convolutional neural network for classification, which competed in the ImageNet Large Scale Visual Recognition Challenge in 2012.
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/caffenet](https://github.com/onnx/models/tree/master/vision/classification/caffenet)
    ///
    /// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
    CaffeNet,
    /// Google's Inception
    Inception(InceptionVersion),
}

/// Google's Inception
#[derive(Debug, Clone)]
pub enum InceptionVersion {
    /// Google's Inception v1
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v1](https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v1)
    ///
    /// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
    V1,
    /// Google's Inception v2
    ///
    /// Source: [https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v2](https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/inception_v2)
    ///
    /// Variant downloaded: ONNX Version 1.4 with Opset Version 9.
    V2,
}

/// ResNet
///
/// Source: [https://github.com/onnx/models/tree/master/vision/classification/resnet](https://github.com/onnx/models/tree/master/vision/classification/resnet)
#[derive(Debug, Clone)]
pub enum ResNet {
    /// ResNet v1
    V1(ResNetV1),
    /// ResNet v2
    V2(ResNetV2),
}
/// ResNet v1
///
/// Source: [https://github.com/onnx/models/tree/master/vision/classification/resnet](https://github.com/onnx/models/tree/master/vision/classification/resnet)
#[derive(Debug, Clone)]
pub enum ResNetV1 {
    /// ResNet18
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet18,
    /// ResNet34
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet34,
    /// ResNet50
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet50,
    /// ResNet101
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet101,
    /// ResNet152
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet152,
}
/// ResNet v2
///
/// Source: [https://github.com/onnx/models/tree/master/vision/classification/resnet](https://github.com/onnx/models/tree/master/vision/classification/resnet)
#[derive(Debug, Clone)]
pub enum ResNetV2 {
    /// ResNet18
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet18,
    /// ResNet34
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet34,
    /// ResNet50
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet50,
    /// ResNet101
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet101,
    /// ResNet152
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    ResNet152,
}

/// ResNet
///
/// Source: [https://github.com/onnx/models/tree/master/vision/classification/resnet](https://github.com/onnx/models/tree/master/vision/classification/resnet)
#[derive(Debug, Clone)]
pub enum Vgg {
    /// VGG with 16 convolutional layers
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    Vgg16,
    /// VGG with 16 convolutional layers, with batch normalization applied after each convolutional layer.
    ///
    /// The batch normalization leads to better convergence and slightly better accuracies.
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    Vgg16Bn,
    /// VGG with 19 convolutional layers
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    Vgg19,
    /// VGG with 19 convolutional layers, with batch normalization applied after each convolutional layer.
    ///
    /// The batch normalization leads to better convergence and slightly better accuracies.
    ///
    /// Variant downloaded: ONNX Version 1.2.1 with Opset Version 7.
    Vgg19Bn,
}

impl ModelUrl for Vision {
    fn fetch_url(&self) -> &'static str {
        match self {
            Vision::ImageClassification(ic) => ic.fetch_url(),
        }
    }
}

impl ModelUrl for ImageClassificationModel {
    fn fetch_url(&self) -> &'static str {
        match self {
            ImageClassificationModel::Mnist => "https://github.com/onnx/models/raw/master/vision/classification/mnist/model/mnist-8.onnx",
            ImageClassificationModel::MobileNet => "https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
            ImageClassificationModel::SqueezeNet => "https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.0-9.onnx",
            ImageClassificationModel::Inception(version) => version.fetch_url(),
            ImageClassificationModel::ResNet(version) => version.fetch_url(),
            ImageClassificationModel::Vgg(variant) => variant.fetch_url(),
            ImageClassificationModel::AlexNet => "https://github.com/onnx/models/raw/master/vision/classification/alexnet/model/bvlcalexnet-9.onnx",
            ImageClassificationModel::GoogleNet => "https://github.com/onnx/models/raw/master/vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.onnx",
            ImageClassificationModel::CaffeNet => "https://github.com/onnx/models/raw/master/vision/classification/caffenet/model/caffenet-9.onnx",
        }
    }
}

impl ModelUrl for InceptionVersion {
    fn fetch_url(&self) -> &'static str {
        match self {
            InceptionVersion::V1 => "https://github.com/onnx/models/raw/master/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-9.onnx",
            InceptionVersion::V2 => "https://github.com/onnx/models/raw/master/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.onnx",
        }
    }
}

impl ModelUrl for ResNet {
    fn fetch_url(&self) -> &'static str {
        match self {
            ResNet::V1(variant) => variant.fetch_url(),
            ResNet::V2(variant) => variant.fetch_url(),
        }
    }
}

impl ModelUrl for ResNetV1 {
    fn fetch_url(&self) -> &'static str {
        match self {
            ResNetV1::ResNet18 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet18-v1-7.onnx",
            ResNetV1::ResNet34 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet34-v1-7.onnx",
            ResNetV1::ResNet50 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v1-7.onnx",
            ResNetV1::ResNet101 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet101-v1-7.onnx",
            ResNetV1::ResNet152 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet152-v1-7.onnx",
        }
    }
}

impl ModelUrl for ResNetV2 {
    fn fetch_url(&self) -> &'static str {
        match self {
            ResNetV2::ResNet18 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet18-v2-7.onnx",
            ResNetV2::ResNet34 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet34-v2-7.onnx",
            ResNetV2::ResNet50 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx",
            ResNetV2::ResNet101 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet101-v2-7.onnx",
            ResNetV2::ResNet152 => "https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet152-v2-7.onnx",
        }
    }
}

impl ModelUrl for Vgg {
    fn fetch_url(&self) -> &'static str {
        match self {
            Vgg::Vgg16 => "https://github.com/onnx/models/raw/master/vision/classification/vgg/model/vgg16-7.onnx",
            Vgg::Vgg16Bn => "https://github.com/onnx/models/raw/master/vision/classification/vgg/model/vgg16-bn-7.onnx",
            Vgg::Vgg19 => "https://github.com/onnx/models/raw/master/vision/classification/vgg/model/vgg19-7.onnx",
            Vgg::Vgg19Bn => "https://github.com/onnx/models/raw/master/vision/classification/vgg/model/vgg19-bn-7.onnx",
        }
    }
}

DOWNLOADABLE_MODELS = {
    (
        "unet",
        "resnest50d",
        "all",
    ): "https://www.googleapis.com/drive/v3/files/1m4QtFZOLg3Pns5HcbSXDskKZn56KkZo5?alt=media&key=AIzaSyAFjU7_3uVx-VGHhp9Kvsda3Su5Ibd_5ys",
    (
        "unet",
        "resnest50d",
        "messidor",
    ): "https://www.googleapis.com/drive/v3/files/1Ob-PKR0dg3KUxsZIFJAtKtboIKph71H9?alt=media&key=AIzaSyAFjU7_3uVx-VGHhp9Kvsda3Su5Ibd_5ys",
    (
        "unet",
        "resnest50d",
        "ddr",
    ): "https://www.googleapis.com/drive/v3/files/1RZKHFc3FjYwtlzEv9o_rFBFL2aUW17uh?alt=media&key=AIzaSyAFjU7_3uVx-VGHhp9Kvsda3Su5Ibd_5ys",
    (
        "unet",
        "resnest50d",
        "retinal_lesions",
    ): "https://www.googleapis.com/drive/v3/files/1MdmfC7O7Tuj9L6FVDN6D8OLutOWqWeKK?alt=media&key=AIzaSyAFjU7_3uVx-VGHhp9Kvsda3Su5Ibd_5ys",
    (
        "unet",
        "resnest50d",
        "fgadr",
    ): "https://www.googleapis.com/drive/v3/files/1q_SmpW8EAK4ppKwrY6-KUyNdnSKKzGZH?alt=media&key=AIzaSyAFjU7_3uVx-VGHhp9Kvsda3Su5Ibd_5ys",
    (
        "unet",
        "resnest50d",
        "idrid",
    ): "https://www.googleapis.com/drive/v3/files/1DI5EcHPkhfl1LdML1jYCxipXYnNhG4Lx?alt=media&key=AIzaSyAFjU7_3uVx-VGHhp9Kvsda3Su5Ibd_5ys",
}


# All current models are trained with warmup dropout, values of 0.2 at the end of training
MODELS_TRAINED_WITH_DROPOUT = list(DOWNLOADABLE_MODELS.keys())

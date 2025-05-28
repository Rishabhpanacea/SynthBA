from monai import transforms
transforms_fn = transforms.Compose([
    transforms.CopyItemsD(keys={'scan_path'}, names=['image']),
    transforms.LoadImageD(keys='image'),
    transforms.EnsureChannelFirstD(keys='image', channel_dim='no_channel'),
    transforms.SpacingD(keys='image', pixdim=1.4),
    transforms.ResizeWithPadOrCropD(keys='image', spatial_size=(130, 130, 130), mode='minimum'),
    transforms.ScaleIntensityD(keys='image', minv=0, maxv=1),
    transforms.Lambda(lambda d: d['image']),
])

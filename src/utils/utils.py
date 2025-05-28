import os
import numpy as np
import nibabel as nib

def get_inputs(inputs_path: str) -> list:
    def get_inputs_from_dir(inputs_path: str) -> list:
        is_nii = lambda f: f.endswith('.nii') or f.endswith('.nii.gz')
        files = os.listdir(inputs_path)
        input_paths = [ os.path.join(inputs_path, f) for f in files if is_nii(f) ]
        return input_paths

    def get_inputs_from_csv(inputs_path: str) -> list:
        with open(inputs_path, 'r') as f:
            return [ p.strip() for p in f.readlines() ]
        
    return get_inputs_from_dir(inputs_path) if os.path.isdir(inputs_path) \
        else get_inputs_from_csv(inputs_path)


def FixMRI(preprocessed_paths):
    for file_path in preprocessed_paths:
        print("inside Fix mri")
        print(file_path)
        img = nib.load(file_path)
        data = img.get_fdata()

        if data.shape[-1] == 1:
            data = np.squeeze(data, axis=-1)
            new_img = nib.Nifti1Image(data, affine=img.affine, header=img.header)
            nib.save(new_img, file_path)
            print(f"Overwritten: {file_path}")
        else:
            print(f"No change needed: {file_path}")
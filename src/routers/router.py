from fastapi import APIRouter, UploadFile,FastAPI
from fastapi.responses import FileResponse
import os
from src.configuration.config import *
import nibabel as nib
import os
from src.utils.utils import *
from src.utils.transformations import transforms_fn
import torch
from torch.utils.data import DataLoader
from monai import transforms
from monai.data import Dataset
from monai.networks.nets.densenet import DenseNet201
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/predict/")
async def create_prediction(file: UploadFile):
    """Predict segmentation from the uploaded NIFTI file."""

    filename = file.filename
    print("file name:-",filename)

    # Construct the full path where the file will be saved
    save_path = os.path.join(InputFolderPath, filename)
    try:
        # Save the uploaded file with the same name as it was inputted
        with open(save_path, 'wb') as tmp:
            data = await file.read()
            tmp.write(data)

        img = nib.load(save_path)
        img = img.get_fdata()
        print(img.shape)
        model_name = "g"
        model_path = os.path.join(checkpointDir, f'synthba-{model_name}.pth')
        assert os.path.exists(model_path), f'Model not found at path {model_path}'
        input_paths = get_inputs(InputFolderPath)

        preprocessed_paths = input_paths
        FixMRI(preprocessed_paths)
        model = DenseNet201(3, 1, 1, dropout_prob=0)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE).eval()
        BatchSize = 1
        data = [ { 'scan_path': p} for p in preprocessed_paths ]
        dataset = Dataset(data=data, transform=transforms_fn)    
        loader = DataLoader(dataset=dataset, batch_size=BatchSize)

        brain_age_list = []
        for i, images in enumerate(loader):
            print(f'processing batch n.{i}')
            with torch.no_grad():
                brain_ages = model(images.to(DEVICE))
                brain_age_list += list(brain_ages.view(-1).cpu().numpy() * 100)
        
        # return str(brain_age_list[0])
        return JSONResponse(content={"predicted_brain_age": float(brain_age_list[0])})

       
        
        # return FileResponse(cleaned_output_files[0], media_type="application/gzip", filename=os.path.basename(cleaned_output_files[0]))

        # return {"message": "success"}
        return "HII"

    finally:
        # os.remove(save_path)
        pass
        # Clean up the temporary file
        # clean_up_temp_file(fd, niftyPath)
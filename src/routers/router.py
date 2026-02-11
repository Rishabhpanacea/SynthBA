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
import uuid
import shutil

router = APIRouter()

@router.post("/predict/")
async def create_prediction(file: UploadFile):
    """Predict segmentation from the uploaded NIFTI file."""

    request_id = str(uuid.uuid4())

    request_input_dir = os.path.join(InputFolderPath, request_id)
    request_strip_dir = os.path.join(preprocess_output_path, "stripped", request_id)
    request_reg_dir = os.path.join(preprocess_output_path, "registered", request_id)
    
    os.makedirs(request_input_dir, exist_ok=True)
    os.makedirs(request_strip_dir, exist_ok=True)
    os.makedirs(request_reg_dir, exist_ok=True)

    filename = file.filename
    print("file name:-",filename)

    # Construct the full path where the file will be saved
    save_path = os.path.join(request_input_dir, filename)

    try:
        with open(save_path, 'wb') as tmp:
            data = await file.read()
            tmp.write(data)

        img = nib.load(save_path)
        img = img.get_fdata()
        print(img.shape)
        model_name = "g"
        model_path = os.path.join(checkpointDir, f'synthba-{model_name}.pth')
        assert os.path.exists(model_path), f'Model not found at path {model_path}'
        input_paths = get_inputs(request_input_dir)

        preprocessed_paths = []
        for inp_path in input_paths:
            
            out_path = preprocess(inp_path, stripping_model_path, request_strip_dir, request_reg_dir, template_path)
            preprocessed_paths.append(out_path)     

        out_path = preprocess(inp_path, stripping_model_path, request_strip_dir, request_reg_dir, template_path)
        
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

        return JSONResponse(content={"predicted_brain_age": float(brain_age_list[0])})

    finally:
        # pass
        shutil.rmtree(request_input_dir, ignore_errors=True)
        shutil.rmtree(request_strip_dir, ignore_errors=True)
        shutil.rmtree(request_reg_dir, ignore_errors=True)
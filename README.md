# Frame Fabric

Frame Fabric is an interactive interior design application that lets you redesign spaces using advanced computer vision pipelines. Select objects to replace furniture items, overlay digital fabrics onto existing structures, and export spatial outputs securely.

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/user/frame_fabric.git
cd frame_fabric
```

### 2. Set Up the Backend
Make sure you have Python 3.10+ installed.
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
# source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Set Up the Frontend
```bash
cd ../frontend
npm install
npm run dev
```

## Assets & Model Setup

### IKEA Dataset
Save your IKEA inventory files to:
`public/assets/ikea_dataset/`

Your structural metadata must be defined inside `public/assets/ikea_dataset/products.json`.

### ML Models
Large weights should reside in the `models/` directory at the root:
* **SAM Weights:** `models/segmentation/sam_vit_b_01ec64.pth`
* **ControlNet/LoRA Weights:** Placed automatically on primary invocations.
#!/usr/bin/env python
import base64
import io
import json
import os
from datetime import datetime
from typing import List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from segmentation import UNet
from classifier import RetinalClassifier, generate_gradcam

def run_segmentation_inference(model, image_path: str, device: torch.device) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
    return mask

def convert_mask_to_color_image(mask: np.ndarray) -> Image.Image:
    palette = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (255, 255, 0),
        3: (0, 255, 0),
    }
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, color in palette.items():
        rgb[mask == k] = color
    return Image.fromarray(rgb)

def image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def calculate_segmentation_metrics(mask: np.ndarray) -> dict:
    total = mask.size
    return {
        "hemorrhages_percentage": float((mask == 1).sum() / total),
        "hard_exudates_percentage": float((mask == 2).sum() / total),
        "microaneurysm_percentage": float((mask == 3).sum() / total),
        "total_abnormality_percentage": float(((mask > 0) & (mask <= 3)).sum() / total),
    }

def json_safe(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return str(obj)

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class RetinalFindings(BaseModel):
    diagnosis: str
    severity_score: int
    key_findings: List[str]
    recommended_actions: List[str]
    confidence_score: float

class ReportGenerator:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=api_key)
        self.parser = PydanticOutputParser(pydantic_object=RetinalFindings)
        self.prompt = PromptTemplate(
            template=(
                "You are an expert ophthalmologist AI assistant.\n"
                "Based on the following analysis results, generate a detailed medical report.\n\n"
                "Classification Results: {classification_result}\n"
                "Segmentation Results: {segmentation_metrics}\n"
                "Confidence Score: {confidence_score}\n"
                "Model Accuracy: {model_accuracy}\n\n"
                "{format_instructions}"
            ),
            input_variables=["classification_result", "segmentation_metrics", "confidence_score", "model_accuracy"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        self.chain = self.prompt | self.llm | self.parser

    def create_report(
        self,
        image_path: str,
        classification_result: dict,
        segmentation_metrics: dict,
        model_accuracy: Optional[float] = None,
        heatmap_path: Optional[str] = None,
        segmentation_model_path: Optional[str] = None,
    ) -> dict:
        findings = self._generate_findings(
            classification_result["class_name"],
            segmentation_metrics,
            classification_result["confidence"],
            model_accuracy,
        )
        report = {
            "report_id": f"RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "date_generated": datetime.now().isoformat(),
            "patient_info": {
                "image_id": os.path.basename(image_path),
                "scan_date": datetime.now().strftime("%Y-%m-%d"),
            },
            "analysis_results": {
                "diagnosis": findings.diagnosis,
                "severity_score": findings.severity_score,
                "confidence_score": findings.confidence_score,
                "key_findings": findings.key_findings,
                "recommended_actions": findings.recommended_actions,
            },
            "technical_metrics": {
                "classification": classification_result,
                "segmentation_metrics": segmentation_metrics,
                "model_accuracy": model_accuracy,
            },
        }
        if segmentation_model_path:
            report["segmentation_output"] = self._segmentation_to_b64(image_path, segmentation_model_path)
        if heatmap_path and os.path.exists(heatmap_path):
            with open(heatmap_path, "rb") as fp:
                report["heatmap"] = base64.b64encode(fp.read()).decode()
        return report

    def save_report(self, report: dict, out_dir: str, image_path: str, heatmap_path: Optional[str] = None) -> str:
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, f"{report['report_id']}.json")
        with open(json_path, "w") as fp:
            json.dump(report, fp, indent=4, default=json_safe)
        html_path = os.path.join(out_dir, f"{report['report_id']}.html")
        with open(html_path, "w") as fp:
            fp.write(self._build_html(report, image_path, heatmap_path))
        return html_path

    def _generate_findings(self, cls: str, seg: dict, conf: float, acc: Optional[float]):
        return self.chain.invoke({
            "classification_result": cls,
            "segmentation_metrics": json.dumps(seg),
            "confidence_score": conf,
            "model_accuracy": f"{acc:.2f}%" if acc is not None else "N/A",
        })

    def _segmentation_to_b64(self, img: str, mdl: str) -> str:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(3, 4).to(dev)
        model.load_state_dict(torch.load(mdl, map_location=dev))
        mask = run_segmentation_inference(model, img, dev)
        return image_to_b64(convert_mask_to_color_image(mask))

    def _build_html(self, rpt: dict, image_path: str, heatmap_path: Optional[str]) -> str:
        # Read the original image and encode in base64
        with open(image_path, "rb") as f:
            original_image = base64.b64encode(f.read()).decode()
        
        # Build the heatmap div if available
        heatmap_div = ""
        if heatmap_path and os.path.exists(heatmap_path):
            with open(heatmap_path, "rb") as f:
                heatmap_image = base64.b64encode(f.read()).decode()
            heatmap_div = f"""
                <div class="image-box">
                    <p><strong>Analysis Heatmap</strong></p>
                    <img src="data:image/png;base64,{heatmap_image}" width="400">
                </div>
            """
        
        # Build key findings and recommended actions lists
        findings_list = "\n".join([f"<div class='finding'>• {finding}</div>" 
                                   for finding in rpt['analysis_results']['key_findings']])
        recommendations_list = "\n".join([f"<div class='finding'>• {rec}</div>" 
                                          for rec in rpt['analysis_results']['recommended_actions']])
        
        # Build segmentation section if segmentation output exists
        segmentation_section = ""
        if "segmentation_output" in rpt:
            segmentation_section = f"""
                <div class="section">
                    <h2>Segmentation Output</h2>
                    <div class="image-container">
                        <div class="image-box">
                            <p><strong>Predicted Segmentation Mask</strong></p>
                            <img src="data:image/png;base64,{rpt['segmentation_output']}" width="400">
                        </div>
                    </div>
                </div>
            """
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced Retinal Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #f4f4f4; color: #333; line-height: 1.6; margin: 0; padding: 20px; }}
                .container {{ max-width: 800px; margin: auto; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); }}
                .header {{ text-align: center; margin-bottom: 20px; }}
                h1 {{ color: #003366; margin: 0; }}
                .section {{ margin: 20px 0; }}
                .section h2 {{ color: #00509E; border-bottom: 2px solid #00509E; padding-bottom: 5px; }}
                .key-findings, .recommendations {{ list-style-type: square; padding-left: 20px; }}
                .image-container {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .image-box {{ text-align: center; }}
                .image-box img {{ width: 350px; border-radius: 8px; border: 2px solid #ddd; }}
                .confidence-score {{ color: #007BFF; font-weight: bold; }}
                .severity-score {{ color: #FF5733; font-weight: bold; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Retinal Analysis Report</h1>
                    <p>Report ID: {rpt['report_id']}</p>
                    <p>Generated on: {rpt['date_generated']}</p>
                </div>
                <div class="section">
                    <h2>Diagnosis Summary</h2>
                    <p><strong>Primary Diagnosis:</strong> {rpt['analysis_results']['diagnosis']}</p>
                    <p><strong>Severity Score:</strong> <span class="severity-score">{rpt['analysis_results']['severity_score']}/4</span></p>
                    <p><strong>Confidence Score:</strong> <span class="confidence-score">{rpt['analysis_results']['confidence_score']:.2%}</span></p>
                </div>
                <div class="section">
                    <h2>Key Findings</h2>
                    {findings_list}
                </div>
                <div class="section">
                    <h2>Recommended Actions</h2>
                    {recommendations_list}
                </div>
                <div class="section">
                    <h2>Visual Analysis</h2>
                    <div class="image-container">
                        <div class="image-box">
                            <p><strong>Original Image</strong></p>
                            <img src="data:image/png;base64,{original_image}" width="400">
                        </div>
                        {heatmap_div}
                    </div>
                </div>
                {segmentation_section}
                <div class="footer">
                    <p>© 2025 Retinal Analysis AI. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html_template

def main():
    image_path = "/Users/inaam/Retinal-Disease-Detection/test_images/0ad36156ad5d.png"
    segmentation_model_path = "/Users/inaam/Retinal-Disease-Detection/retinal_segmentation_model.pth"
    classifier_model_path = "/Users/inaam/Retinal-Disease-Detection/Best Model.pth"
    openai_api_key = "OPENAI_KEY"
    output_dir = "./reports"
    generate_heatmap = False
    classifier = RetinalClassifier(num_classes=5)
    classifier.load_model(classifier_model_path)
    classification_result = classifier.predict_single_image(image_path)
    accuracy = None
    try:
        from transformers.models.vit.configuration_vit import ViTConfig
        import torch.serialization as _ts
        _ts.add_safe_globals([ViTConfig])
        ckpt = torch.load(classifier_model_path, map_location="cpu", weights_only=True)
        for key in ("best_acc", "accuracy", "val_accuracy"):
            if isinstance(ckpt, dict) and key in ckpt:
                accuracy = ckpt[key] * 100 if ckpt[key] <= 1 else ckpt[key]
                break
    except Exception as e:
        print("Could not extract stored accuracy:", e)
    print(f"Diagnosis  : {classification_result['class_name']}")
    print(f"Confidence : {classification_result['confidence']:.4f}")
    if accuracy is not None:
        print(f"Model Acc. : {accuracy:.2f}%")
    heatmap_path = None
    if generate_heatmap:
        try:
            heatmap = generate_gradcam(classifier, image_path)
            heatmap_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_heatmap.png")
            os.makedirs(output_dir, exist_ok=True)
            heatmap.save(heatmap_path)
        except Exception as e:
            print("Heat‑map generation failed:", e)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model = UNet(n_channels=3, n_classes=4).to(device)
    seg_model.load_state_dict(torch.load(segmentation_model_path, map_location=device))
    mask = run_segmentation_inference(seg_model, image_path, device)
    seg_metrics = calculate_segmentation_metrics(mask)
    generator = ReportGenerator(openai_api_key)
    report = generator.create_report(
        image_path=image_path,
        classification_result=classification_result,
        segmentation_metrics=seg_metrics,
        model_accuracy=accuracy,
        heatmap_path=heatmap_path,
        segmentation_model_path=segmentation_model_path,
    )
    html_path = generator.save_report(report, output_dir, image_path, heatmap_path)
    print(f"Report saved to {html_path}")

if __name__ == "__main__":
    main()

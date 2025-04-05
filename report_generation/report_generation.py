#!/usr/bin/env python
import json
import os
import time
from datetime import datetime
import io
import base64
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from segmentation import UNet

def run_segmentation_inference(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        pred_mask = torch.argmax(probabilities, dim=1).squeeze().cpu().numpy()
    return pred_mask

def convert_mask_to_color_image(mask: np.ndarray) -> Image.Image:
    colors = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (255, 255, 0),
        3: (0, 255, 0)
    }
    h, w = mask.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in colors.items():
        color_image[mask == cls] = color
    return Image.fromarray(color_image)

def image_to_base64_str(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from typing import List, Optional

class RetinalFindings(BaseModel):
    diagnosis: str = Field(description="Primary diagnosis based on the image analysis")
    severity_score: int = Field(description="Severity score from 0-4")
    key_findings: List[str] = Field(description="List of specific abnormalities or findings detected")
    recommended_actions: List[str] = Field(description="Recommended follow-up actions or treatments")
    confidence_score: float = Field(description="Model's confidence in the diagnosis (0-1)")

class ReportGenerator:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4",
            openai_api_key=openai_api_key
        )
        self.parser = PydanticOutputParser(pydantic_object=RetinalFindings)
        self.prompt_template = PromptTemplate(
            template="""You are an expert ophthalmologist AI assistant. Based on the following analysis results, 
generate a detailed medical report. Please format your response according to the specified output format.

Classification Results: {classification_result}
Segmentation Results: {segmentation_metrics}
Confidence Score: {confidence_score}

Rules for report generation:
1. Be specific about the findings and their implications
2. Use professional medical terminology
3. Provide clear severity assessment
4. Include actionable recommendations

{format_instructions}
""",
            input_variables=["classification_result", "segmentation_metrics", "confidence_score"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chain = self.prompt_template | self.llm | self.parser

    def generate_findings(self, classification_result: str, segmentation_metrics: dict, 
                         confidence_score: float) -> RetinalFindings:
        result = self.chain.invoke({
            "classification_result": classification_result,
            "segmentation_metrics": json.dumps(segmentation_metrics),
            "confidence_score": confidence_score
        })
        return result

    def create_report(self, image_path: str, classification_result: str, 
                     segmentation_metrics: dict, confidence_score: float,
                     heatmap_path: Optional[str] = None,
                     segmentation_model_path: Optional[str] = None) -> dict:
        findings = self.generate_findings(
            classification_result,
            segmentation_metrics,
            confidence_score
        )
        report = {
            "report_id": f"RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "date_generated": datetime.now().isoformat(),
            "patient_info": {
                "image_id": os.path.basename(image_path),
                "scan_date": datetime.now().strftime("%Y-%m-%d")
            },
            "analysis_results": {
                "diagnosis": findings.diagnosis,
                "severity_score": findings.severity_score,
                "confidence_score": findings.confidence_score,
                "key_findings": findings.key_findings,
                "recommended_actions": findings.recommended_actions
            },
            "technical_metrics": {
                "classification": classification_result,
                "segmentation_metrics": segmentation_metrics
            }
        }
        if segmentation_model_path:
            seg_output_b64 = self._run_segmentation_and_convert(image_path, segmentation_model_path)
            report["segmentation_output"] = seg_output_b64
        return report

    def _run_segmentation_and_convert(self, image_path: str, model_path: str) -> str:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet(n_channels=3, n_classes=4).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        pred_mask = run_segmentation_inference(model, image_path, device)
        color_image = convert_mask_to_color_image(pred_mask)
        seg_output_b64 = image_to_base64_str(color_image)
        return seg_output_b64

    def save_report(self, report: dict, output_dir: str, image_path: str, 
                   heatmap_path: Optional[str] = None):
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, f"{report['report_id']}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        html_report = self._generate_html_report(report, image_path, heatmap_path)
        html_path = os.path.join(output_dir, f"{report['report_id']}.html")
        with open(html_path, 'w') as f:
            f.write(html_report)

    def _generate_html_report(self, report: dict, image_path: str, 
                        heatmap_path: Optional[str] = None) -> str:
        html_template = """
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
                    <p>Report ID: {report_id}</p>
                    <p>Generated on: {date_generated}</p>
                </div>
                <div class="section">
                    <h2>Diagnosis Summary</h2>
                    <p><strong>Primary Diagnosis:</strong> {diagnosis}</p>
                    <p><strong>Severity Score:</strong> <span class="severity-score">{severity}/4</span></p>
                    <p><strong>Confidence Score:</strong> <span class="confidence-score">{confidence:.2%}</span></p>
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
        with open(image_path, "rb") as image_file:
            original_image = base64.b64encode(image_file.read()).decode()
        heatmap_div = ""
        if heatmap_path:
            with open(heatmap_path, "rb") as heatmap_file:
                heatmap_image = base64.b64encode(heatmap_file.read()).decode()
                heatmap_div = f"""
                <div>
                    <p><strong>Analysis Heatmap</strong></p>
                    <img src="data:image/png;base64,{heatmap_image}" width="400">
                </div>
                """
        findings_list = "\n".join([f"<div class='finding'>• {finding}</div>" 
                                 for finding in report['analysis_results']['key_findings']])
        recommendations_list = "\n".join([f"<div class='finding'>• {rec}</div>" 
                                        for rec in report['analysis_results']['recommended_actions']])
        segmentation_section = ""
        if "segmentation_output" in report:
            segmentation_section = f"""
            <div class="section">
                <h2>Segmentation Output</h2>
                <div class="image-container">
                    <div class="image-box">
                        <p><strong>Predicted Segmentation Mask</strong></p>
                        <img src="data:image/png;base64,{report['segmentation_output']}" width="400">
                    </div>
                </div>
            </div>
            """
        html_report = html_template.format(
            report_id=report['report_id'],
            date_generated=report['date_generated'],
            diagnosis=report['analysis_results']['diagnosis'],
            severity=report['analysis_results']['severity_score'],
            confidence=report['analysis_results']['confidence_score'],
            findings_list=findings_list,
            recommendations_list=recommendations_list,
            original_image=original_image,
            heatmap_div=heatmap_div,
            segmentation_section=segmentation_section
        )
        return html_report

if __name__ == "__main__":
    image_path = "/Users/inaam/Retinal-Disease-Detection/test_images/0ad36156ad5d.png"
    segmentation_model_path = "/Users/inaam/Retinal-Disease-Detection/retinal_segmentation_model.pth"
    heatmap_path = None
    output_dir = "./reports"
    classification_result = "Moderate Non-proliferative Diabetic Retinopathy"
    confidence_score = 0.92
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_instance = UNet(n_channels=3, n_classes=4).to(device)
    model_instance.load_state_dict(torch.load(segmentation_model_path, map_location=device))
    model_instance.eval()
    pred_mask = run_segmentation_inference(model_instance, image_path, device)
    total_pixels = pred_mask.size
    lesion_area_percentage = (pred_mask == 1).sum() / total_pixels
    segmentation_metrics = {
        "lesion_area_percentage": lesion_area_percentage
    }
    report_generator = ReportGenerator(openai_api_key="")
    report = report_generator.create_report(
        image_path=image_path,
        classification_result=classification_result,
        segmentation_metrics=segmentation_metrics,
        confidence_score=confidence_score,
        heatmap_path=heatmap_path,
        segmentation_model_path=segmentation_model_path
    )
    report_generator.save_report(
        report=report,
        output_dir=output_dir,
        image_path=image_path,
        heatmap_path=heatmap_path
    )
    print(f"Report generated and saved in {output_dir}")

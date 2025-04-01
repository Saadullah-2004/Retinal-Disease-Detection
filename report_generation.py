from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import json
from datetime import datetime
import os
from PIL import Image
import numpy as np

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
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def generate_findings(self, classification_result: str, segmentation_metrics: dict, 
                         confidence_score: float) -> RetinalFindings:
        """Generate structured findings from model outputs"""
        result = self.chain.run(
            classification_result=classification_result,
            segmentation_metrics=json.dumps(segmentation_metrics),
            confidence_score=confidence_score
        )
        return self.parser.parse(result)

    def create_report(self, image_path: str, classification_result: str, 
                     segmentation_metrics: dict, confidence_score: float,
                     heatmap_path: Optional[str] = None) -> dict:
        """
        Create a complete medical report including findings and visualizations
        
        Args:
            image_path: Path to the original retinal image
            classification_result: Model's classification output
            segmentation_metrics: Dictionary containing segmentation analysis metrics
            confidence_score: Model's confidence score
            heatmap_path: Optional path to the explanation heatmap
        
        Returns:
            Dictionary containing the complete report
        """
        # Generate findings using LangChain
        findings = self.generate_findings(
            classification_result,
            segmentation_metrics,
            confidence_score
        )
        
        # Create report structure
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
        
        return report

    def save_report(self, report: dict, output_dir: str, image_path: str, 
                   heatmap_path: Optional[str] = None):
        """
        Save the report and associated visualizations
        
        Args:
            report: Generated report dictionary
            output_dir: Directory to save the report
            image_path: Path to the original image
            heatmap_path: Optional path to the explanation heatmap
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON report
        report_path = os.path.join(output_dir, f"{report['report_id']}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Generate HTML report
        html_report = self._generate_html_report(report, image_path, heatmap_path)
        html_path = os.path.join(output_dir, f"{report['report_id']}.html")
        with open(html_path, 'w') as f:
            f.write(html_report)

    def _generate_html_report(self, report: dict, image_path: str, 
                        heatmap_path: Optional[str] = None) -> str:
        """Generate an HTML version of the report"""
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
                <div class="footer">
                    <p>© 2025 Retinal Analysis AI. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Convert images to base64
        with open(image_path, "rb") as image_file:
            import base64
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
        
        # Generate findings and recommendations lists
        findings_list = "\n".join([f"<div class='finding'>• {finding}</div>" 
                                 for finding in report['analysis_results']['key_findings']])
        recommendations_list = "\n".join([f"<div class='finding'>• {rec}</div>" 
                                        for rec in report['analysis_results']['recommended_actions']])
        
        # Fill template
        html_report = html_template.format(
            report_id=report['report_id'],
            date_generated=report['date_generated'],
            diagnosis=report['analysis_results']['diagnosis'],
            severity=report['analysis_results']['severity_score'],
            confidence=report['analysis_results']['confidence_score'],
            findings_list=findings_list,
            recommendations_list=recommendations_list,
            original_image=original_image,
            heatmap_div=heatmap_div
        )
        
        return html_report

# Usage example
if __name__ == "__main__":
    # Initialize the report generator
    report_generator = ReportGenerator(openai_api_key="")
    
    # Example data
    classification_result = "Moderate Non-proliferative Diabetic Retinopathy"
    segmentation_metrics = {
        "microaneurysms_count": 12,
        "hemorrhages_count": 3,
        "exudates_area_percentage": 0.05
    }
    confidence_score = 0.92
    
    # Generate and save report
    report = report_generator.create_report(
        image_path="/Users/devshah/Documents/WorkSpace/University/year 3/CSC490/Zero-Shot-Object-Tracking-FPS/APTOS 2019 Blindness Detection/test_images/0a2b5e1a0be8.png",
        classification_result=classification_result,
        segmentation_metrics=segmentation_metrics,
        confidence_score=confidence_score,
        heatmap_path="/Users/devshah/Documents/WorkSpace/University/year 3/CSC490/Zero-Shot-Object-Tracking-FPS/APTOS 2019 Blindness Detection/test_images/0a2b5e1a0be8.png"
    )
    
    # Save report files
    report_generator.save_report(
        report=report,
        output_dir="./reports",
        image_path="/Users/devshah/Documents/WorkSpace/University/year 3/CSC490/Zero-Shot-Object-Tracking-FPS/APTOS 2019 Blindness Detection/test_images/0a2b5e1a0be8.png",
        heatmap_path="/Users/devshah/Documents/WorkSpace/University/year 3/CSC490/Zero-Shot-Object-Tracking-FPS/APTOS 2019 Blindness Detection/test_images/0a2b5e1a0be8.png"
    )

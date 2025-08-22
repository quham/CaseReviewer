"""
AI Insights Service for CaseReviewer
Provides sophisticated analysis and personalized recommendations for social workers
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class AIInsightsService:
    """Service for generating AI-powered insights from case reviews"""
    
    def __init__(self):
        self.risk_keywords = {
            "neglect": ["neglect", "neglected", "neglecting", "basic needs", "supervision", "care"],
            "abuse": ["abuse", "abused", "abusive", "physical abuse", "emotional abuse", "sexual abuse"],
            "domestic_violence": ["domestic violence", "domestic abuse", "intimate partner violence", "family violence"],
            "mental_health": ["mental health", "depression", "anxiety", "psychosis", "suicidal", "self-harm"],
            "substance_abuse": ["substance abuse", "drugs", "alcohol", "addiction", "intoxicated"],
            "poverty": ["poverty", "homeless", "eviction", "financial hardship", "unemployment"],
            "educational": ["school", "education", "truancy", "academic", "learning difficulties"],
            "medical": ["medical", "health", "illness", "disability", "chronic condition"]
        }
        
        self.intervention_keywords = {
            "safety_planning": ["safety plan", "safety planning", "risk assessment", "protective measures"],
            "family_support": ["family support", "parenting support", "family therapy", "counseling"],
            "multi_agency": ["multi-agency", "intervention", "coordination", "partnership"],
            "legal": ["legal", "court", "order", "proceedings", "custody"],
            "medical": ["medical", "health", "treatment", "therapy", "medication"],
            "educational": ["education", "school", "tutoring", "special needs", "IEP"]
        }
        
        self.outcome_patterns = {
            "positive": ["improved", "better", "successful", "resolved", "stable", "recovery"],
            "negative": ["worsened", "deteriorated", "failed", "unsuccessful", "relapse"],
            "ongoing": ["ongoing", "continuing", "monitoring", "follow-up", "review"]
        }
    
    def analyze_case_complexity(self, case_data: dict) -> Dict[str, Any]:
        """Analyze the complexity of a case based on multiple factors"""
        complexity_score = 0
        complexity_factors = []
        
        # Risk type complexity
        risk_types = case_data.get("risk_types", [])
        if len(risk_types) > 3:
            complexity_score += 30
            complexity_factors.append("Multiple risk types")
        elif len(risk_types) > 1:
            complexity_score += 20
            complexity_factors.append("Several risk types")
        
        # Agency involvement complexity
        agencies = case_data.get("agencies", [])
        if len(agencies) > 4:
            complexity_score += 25
            complexity_factors.append("High multi-agency involvement")
        elif len(agencies) > 2:
            complexity_score += 15
            complexity_factors.append("Multi-agency involvement")
        
        # Timeline complexity
        timeline_events = case_data.get("timeline_events", [])
        if len(timeline_events) > 10:
            complexity_score += 20
            complexity_factors.append("Complex timeline with many events")
        elif len(timeline_events) > 5:
            complexity_score += 10
            complexity_factors.append("Moderate timeline complexity")
        
        # Content complexity
        content = case_data.get("content", "")
        if len(content) > 5000:
            complexity_score += 15
            complexity_factors.append("Extensive case documentation")
        
        # Determine complexity level
        if complexity_score >= 70:
            complexity_level = "Very High"
        elif complexity_score >= 50:
            complexity_level = "High"
        elif complexity_score >= 30:
            complexity_level = "Medium"
        else:
            complexity_level = "Low"
        
        return {
            "complexity_score": complexity_score,
            "complexity_level": complexity_level,
            "complexity_factors": complexity_factors,
            "risk_type_count": len(risk_types),
            "agency_count": len(agencies),
            "timeline_event_count": len(timeline_events)
        }
    
    def extract_risk_assessment(self, case_data: dict) -> Dict[str, Any]:
        """Extract and analyze risk assessment from case data"""
        risk_assessment = {
            "overall_risk_level": "Unknown",
            "risk_factors": [],
            "protective_factors": [],
            "risk_indicators": [],
            "urgency_level": "Unknown"
        }
        
        # Analyze risk types
        risk_types = case_data.get("risk_types", [])
        risk_factors = case_data.get("risk_factors", [])
        warning_signs = case_data.get("warning_signs_early", [])
        
        # Determine overall risk level
        high_risk_types = ["abuse", "domestic_violence", "sexual_abuse"]
        medium_risk_types = ["neglect", "mental_health", "substance_abuse"]
        
        high_risk_count = sum(1 for rt in risk_types if any(hr in rt.lower() for hr in high_risk_types))
        medium_risk_count = sum(1 for rt in risk_types if any(mr in rt.lower() for mr in medium_risk_types))
        
        if high_risk_count > 0:
            risk_assessment["overall_risk_level"] = "High"
            risk_assessment["urgency_level"] = "Immediate"
        elif medium_risk_count > 0 or len(risk_factors) > 2:
            risk_assessment["overall_risk_level"] = "Medium"
            risk_assessment["urgency_level"] = "High"
        elif len(risk_types) > 0:
            risk_assessment["overall_risk_level"] = "Low"
            risk_assessment["urgency_level"] = "Medium"
        
        # Extract risk factors
        risk_assessment["risk_factors"] = risk_factors[:5]
        risk_assessment["risk_indicators"] = warning_signs[:5]
        
        # Identify protective factors (opposite of risk factors)
        content = case_data.get("content", "").lower()
        protective_keywords = ["support", "family", "stable", "improved", "positive", "strength"]
        protective_factors = []
        
        for keyword in protective_keywords:
            if keyword in content:
                protective_factors.append(f"Evidence of {keyword}")
        
        risk_assessment["protective_factors"] = protective_factors[:3]
        
        return risk_assessment
    
    def generate_intervention_strategies(self, case_data: dict, query_context: str = "") -> Dict[str, Any]:
        """Generate evidence-based intervention strategies"""
        strategies = {
            "immediate_actions": [],
            "short_term_interventions": [],
            "long_term_strategies": [],
            "multi_agency_coordination": [],
            "monitoring_requirements": []
        }
        
        risk_types = [rt.lower() for rt in case_data.get("risk_types", [])]
        barriers = case_data.get("barriers", [])
        agencies = case_data.get("agencies", [])
        
        # Immediate actions based on risk types
        if any("abuse" in rt for rt in risk_types):
            strategies["immediate_actions"].extend([
                "Immediate safety planning for child and family",
                "Risk assessment and safety evaluation",
                "Contact relevant authorities if required"
            ])
        
        if any("domestic_violence" in rt for rt in risk_types):
            strategies["immediate_actions"].extend([
                "Safety planning with domestic violence specialist",
                "Risk assessment for family members",
                "Coordination with police and domestic violence services"
            ])
        
        if any("neglect" in rt for rt in risk_types):
            strategies["immediate_actions"].extend([
                "Assessment of basic needs provision",
                "Home visit and family assessment",
                "Coordination with health and education services"
            ])
        
        # Short-term interventions
        if "mental_health" in risk_types:
            strategies["short_term_interventions"].extend([
                "Mental health assessment and referral",
                "Crisis intervention if needed",
                "Family mental health support"
            ])
        
        if "substance_abuse" in risk_types:
            strategies["short_term_interventions"].extend([
                "Substance abuse assessment",
                "Referral to addiction services",
                "Family support for substance-related issues"
            ])
        
        # Long-term strategies
        strategies["long_term_strategies"].extend([
            "Regular case review and monitoring",
            "Family support and parenting programs",
            "Multi-agency case coordination",
            "Regular risk assessments and updates"
        ])
        
        # Multi-agency coordination
        if agencies:
            strategies["multi_agency_coordination"].extend([
                f"Regular coordination meetings with {', '.join(agencies[:3])}",
                "Shared case management and information sharing",
                "Joint intervention planning and review"
            ])
        
        # Monitoring requirements
        strategies["monitoring_requirements"].extend([
            "Regular home visits and family contact",
            "Progress reviews every 3-6 months",
            "Risk assessment updates",
            "Multi-agency case review meetings"
        ])
        
        return strategies
    
    def create_personalized_summary(self, case_data: dict, query: str, user_role: str = "social_worker") -> Dict[str, Any]:
        """Create a personalized summary based on user role and search query"""
        summary = {
            "executive_summary": "",
            "key_findings": [],
            "professional_insights": [],
            "action_items": [],
            "learning_points": []
        }
        
        # Create executive summary
        title = case_data.get("title", "Case Review")
        outcome = case_data.get("outcome", "Outcome not specified")
        risk_level = self.extract_risk_assessment(case_data)["overall_risk_level"]
        
        summary["executive_summary"] = f"{title}. This case involves {risk_level.lower()} risk factors with {outcome.lower()}. "
        
        if case_data.get("barriers"):
            summary["executive_summary"] += f"Key barriers identified include {', '.join(case_data['barriers'][:2])}."
        
        # Key findings
        if case_data.get("warning_signs_early"):
            summary["key_findings"].extend([
                f"Early warning signs: {', '.join(case_data['warning_signs_early'][:3])}"
            ])
        
        if case_data.get("risk_factors"):
            summary["key_findings"].extend([
                f"Risk factors: {', '.join(case_data['risk_factors'][:3])}"
            ])
        
        # Professional insights based on role
        if user_role == "social_worker":
            summary["professional_insights"].extend([
                "Consider regular home visits and family support",
                "Monitor for early warning signs and risk factors",
                "Maintain multi-agency coordination and communication"
            ])
        elif user_role == "manager":
            summary["professional_insights"].extend([
                "Ensure adequate resource allocation for complex cases",
                "Monitor case progression and intervention effectiveness",
                "Support staff with regular supervision and training"
            ])
        
        # Action items
        strategies = self.generate_intervention_strategies(case_data)
        summary["action_items"].extend(strategies["immediate_actions"][:3])
        
        # Learning points
        if case_data.get("barriers"):
            summary["learning_points"].extend([
                f"Barrier to address: {barrier}" for barrier in case_data["barriers"][:2]
            ])
        
        if case_data.get("outcome"):
            summary["learning_points"].append(f"Outcome: {case_data['outcome']}")
        
        return summary
    
    def extract_timeline_insights(self, timeline_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract insights from timeline events"""
        timeline_analysis = {
            "critical_events": [],
            "intervention_points": [],
            "response_times": [],
            "escalation_patterns": [],
            "coordination_events": []
        }
        
        if not timeline_events:
            return timeline_analysis
        
        # Sort events by date
        sorted_events = sorted(timeline_events, key=lambda x: x.get("event_date", ""))
        
        # Identify critical events
        critical_keywords = ["crisis", "emergency", "incident", "allegation", "referral"]
        for event in sorted_events:
            event_type = event.get("event_type", "").lower()
            description = event.get("description", "").lower()
            
            if any(keyword in event_type or keyword in description for keyword in critical_keywords):
                timeline_analysis["critical_events"].append({
                    "date": event.get("event_date"),
                    "type": event.get("event_type"),
                    "description": event.get("description")
                })
        
        # Identify intervention points
        intervention_keywords = ["intervention", "assessment", "plan", "review", "meeting"]
        for event in sorted_events:
            event_type = event.get("event_type", "").lower()
            if any(keyword in event_type for keyword in intervention_keywords):
                timeline_analysis["intervention_points"].append({
                    "date": event.get("event_date"),
                    "type": event.get("event_type"),
                    "track": event.get("track")
                })
        
        # Analyze response times between critical events and interventions
        critical_dates = [event["date"] for event in timeline_analysis["critical_events"]]
        intervention_dates = [event["date"] for event in timeline_analysis["intervention_points"]]
        
        if critical_dates and intervention_dates:
            # Simple response time analysis (could be enhanced with actual date parsing)
            timeline_analysis["response_times"].append("Response time analysis available")
        
        return timeline_analysis
    
    def generate_case_recommendations(self, case_data: dict, query_context: str = "") -> List[str]:
        """Generate specific, actionable recommendations"""
        recommendations = []
        
        # Risk-based recommendations
        risk_types = [rt.lower() for rt in case_data.get("risk_types", [])]
        
        if any("abuse" in rt for rt in risk_types):
            recommendations.extend([
                "Implement immediate safety planning with family",
                "Conduct comprehensive risk assessment",
                "Establish regular safety checks and monitoring"
            ])
        
        if any("neglect" in rt for rt in risk_types):
            recommendations.extend([
                "Assess and address basic needs provision",
                "Develop family support and parenting skills program",
                "Establish regular home visit schedule"
            ])
        
        if any("domestic_violence" in rt for rt in risk_types):
            recommendations.extend([
                "Coordinate with domestic violence specialists",
                "Develop safety planning for all family members",
                "Establish emergency contact protocols"
            ])
        
        # Barrier-based recommendations
        barriers = case_data.get("barriers", [])
        for barrier in barriers[:3]:
            recommendations.append(f"Address barrier: {barrier}")
        
        # Agency coordination recommendations
        agencies = case_data.get("agencies", [])
        if len(agencies) > 2:
            recommendations.append("Establish regular multi-agency case coordination meetings")
            recommendations.append("Develop shared case management protocols")
        
        # Monitoring and review recommendations
        recommendations.extend([
            "Conduct regular case reviews every 3-6 months",
            "Update risk assessments based on new information",
            "Maintain detailed case documentation and progress notes"
        ])
        
        return recommendations[:8]  # Limit to top 8 recommendations

# Initialize the service
ai_insights_service = AIInsightsService()

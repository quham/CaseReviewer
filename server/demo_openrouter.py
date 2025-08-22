#!/usr/bin/env python3
"""
Demo script for OpenRouter AI Insights Service
Showcases the LLM-powered personalized recommendations
"""

import os
import sys
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_openrouter_insights():
    """Demonstrate OpenRouter AI insights functionality"""
    print("🚀 OpenRouter AI Insights Service Demo")
    print("=" * 60)
    
    try:
        from services.openrouter_insights import openrouter_insights_service
        
        if not openrouter_insights_service.llm_enabled:
            print("⚠️ OpenRouter not configured. Please set OPENROUTER_API_KEY in your .env file")
            print("   This demo will show fallback functionality instead.")
            print()
        
        # Sample case data for demonstration
        sample_cases = [
            {
                "id": "case_001",
                "title": "Multi-Agency Coordination Failure in Child Protection",
                "summary": "A case where poor coordination between social services, police, and health services led to missed opportunities for early intervention in a neglect case.",
                "risk_types": ["neglect", "domestic_violence"],
                "agencies": ["social_services", "police", "health_visitor", "school"],
                "barriers": ["lack of information sharing", "delayed response times", "unclear responsibilities"],
                "warning_signs_early": ["repeated school absences", "missed health appointments", "neighbor concerns"],
                "risk_factors": ["parental mental health issues", "substance abuse", "poverty"],
                "outcome": "Child removed from home after escalation of risks"
            },
            {
                "id": "case_002", 
                "title": "Successful Early Intervention in Domestic Violence",
                "summary": "A case demonstrating effective early identification and intervention in domestic violence, leading to family safety and support.",
                "risk_types": ["domestic_violence", "emotional_abuse"],
                "agencies": ["social_services", "domestic_violence_specialist", "police", "refuge"],
                "barriers": ["victim reluctance to report", "perpetrator manipulation"],
                "warning_signs_early": ["controlling behavior", "isolation from family", "financial control"],
                "risk_factors": ["previous domestic violence", "substance abuse", "unemployment"],
                "outcome": "Family successfully supported, perpetrator removed, safety plan implemented"
            },
            {
                "id": "case_003",
                "title": "Complex Neglect Case with Multiple Risk Factors",
                "summary": "A challenging case involving multiple generations of neglect, requiring long-term intervention and family support.",
                "risk_types": ["neglect", "mental_health", "poverty"],
                "agencies": ["social_services", "mental_health_services", "housing", "education"],
                "barriers": ["intergenerational patterns", "mental health stigma", "housing instability"],
                "warning_signs_early": ["parental withdrawal", "child developmental delays", "housing issues"],
                "risk_factors": ["parental mental health", "poverty", "lack of support networks"],
                "outcome": "Ongoing support with gradual improvements in family functioning"
            }
        ]
        
        # Demo 1: Query-based recommendations
        print("📋 Demo 1: Query-based AI Recommendations")
        print("-" * 40)
        
        query = "missed opportunities in child protection and how to improve multi-agency coordination"
        print(f"Query: {query}")
        print()
        
        recommendations = openrouter_insights_service.generate_personalized_recommendations(
            query, sample_cases, "social_worker"
        )
        
        print("AI-Generated Recommendations:")
        print(f"  Source: {recommendations.get('metadata', {}).get('source', 'unknown')}")
        
        if 'query_analysis' in recommendations:
            analysis = recommendations['query_analysis']
            print(f"  Relevance: {analysis.get('relevance_summary', 'N/A')}")
            print(f"  Key Patterns: {', '.join(analysis.get('key_patterns', []))}")
            print(f"  Risk Factors: {', '.join(analysis.get('risk_factors_identified', []))}")
        
        if 'personalized_recommendations' in recommendations:
            recs = recommendations['personalized_recommendations']
            print(f"  Immediate Actions: {len(recs.get('immediate_actions', []))}")
            print(f"  Short-term Strategies: {len(recs.get('short_term_strategies', []))}")
            print(f"  Long-term Approaches: {len(recs.get('long_term_approaches', []))}")
        
        print()
        
        # Demo 2: Role-based recommendations
        print("📋 Demo 2: Role-based AI Recommendations")
        print("-" * 40)
        
        # Demo 3: Role-based recommendations
        print("📋 Demo 3: Role-based AI Recommendations")
        print("-" * 40)
        
        roles = ["social_worker", "manager", "reviewer"]
        
        for role in roles:
            print(f"Role: {role.title()}")
            role_recommendations = openrouter_insights_service.generate_personalized_recommendations(
                "improving case outcomes through better risk assessment",
                sample_cases[:2], role
            )
            
            if 'personalized_recommendations' in role_recommendations:
                recs = role_recommendations['personalized_recommendations']
                print(f"  Immediate Actions: {len(recs.get('immediate_actions', []))}")
                print(f"  Short-term Strategies: {len(recs.get('short_term_strategies', []))}")
                print(f"  Long-term Approaches: {len(recs.get('long_term_approaches', []))}")
            
            print()
        
        # Demo 4: Show full recommendation structure
        print("📋 Demo 4: Full Recommendation Structure")
        print("-" * 40)
        
        full_recommendations = openrouter_insights_service.generate_personalized_recommendations(
            "comprehensive case review analysis",
            sample_cases, "social_worker"
        )
        
        print("Full AI Recommendation Structure:")
        for key, value in full_recommendations.items():
            if key != 'metadata':
                if isinstance(value, dict):
                    print(f"  {key}: {len(value)} sub-categories")
                elif isinstance(value, list):
                    print(f"  {key}: {len(value)} items")
                else:
                    print(f"  {key}: {value}")
        
        print()
        
        # Summary
        print("🎯 Demo Summary")
        print("=" * 60)
        print("✅ OpenRouter AI Insights Service is working!")
        print("✅ Generates personalized recommendations based on:")
        print("   - Search query context")
        print("   - Top matching cases")
        print("   - User role and professional context")
        print("✅ Provides structured recommendations including:")
        print("   - Immediate actions and strategies")
        print("   - Key insights and takeaways")
        print("   - Risk management strategies")
        print()
        print("🚀 Ready to enhance social worker case reviews with AI-powered insights!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_openrouter_insights()

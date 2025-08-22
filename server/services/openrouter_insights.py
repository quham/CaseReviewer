"""
OpenRouter AI Insights Service for CaseReviewer
Provides LLM-powered personalized recommendations and insights for social workers
"""

import os
import logging
import requests
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class OpenRouterInsightsService:
    """Service for generating LLM-powered insights using OpenRouter"""
    
    def __init__(self):
        load_dotenv()
        
        # OpenRouter Configuration
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.base_url = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        # self.model = os.getenv('OPENROUTER_MODEL', 'meta-llama/llama-3.3-70b-instruct:free')
        
        # Model fallback list in order of preference
        self.models = [
            "meta-llama/llama-3.3-70b-instruct:free",
            "deepseek/deepseek-chat-v3-0324:free",
            "deepseek/deepseek-r1-0528:free",
            "google/gemini-2.0-flash-exp:free",
            "anthropic/claude-3-haiku:free",
            "qwen/qwen3-8b:free",
            "deepseek/deepseek-r1-0528-qwen3-8b:free",
            "mistralai/mistral-7b-instruct:free"
        ]
        
        if not self.api_key:
            logger.warning("⚠️ OPENROUTER_API_KEY not found. LLM insights will be disabled.")
            self.llm_enabled = False
        else:
            self.llm_enabled = True
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/casereviewer",
                "X-Title": "CaseReviewer AI Insights"
            }
            logger.info("✅ OpenRouter LLM service initialized")
            logger.info(f"🔄 Fallback models configured: {len(self.models)} models")
            logger.info(f"🔝 First fallback: {self.models[0] if self.models else 'None'}")
    
    def _call_openrouter_single_model(self, prompt: str, max_tokens: int = 2000, model: str = None) -> Optional[Tuple[str, str]]:
        """Make a call to OpenRouter API with a specific model"""
        if not self.llm_enabled:
            return None
        
        if not model:
            model = self.models[0] if self.models else None
            
        if not model:
            return None
        
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,  # Lower temperature for more consistent outputs
                "max_tokens": max_tokens,
                "top_p": 0.9
            }
        
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                logger.info(f"✅ Successfully used model: {model}")
                return content.strip(), model
            else:
                logger.warning(f"⚠️ Model {model} failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Model {model} error: {e}")
            return None

    def _call_openrouter(self, prompt: str, max_tokens: int = 2000) -> Optional[Tuple[str, str]]:
        """Make a call to OpenRouter API - tries models in order until one works"""
        if not self.llm_enabled:
            return None
        
        # Try models in order until one works
        for model in self.models:
            result = self._call_openrouter_single_model(prompt, max_tokens, model)
            if result:
                return result
        
        logger.error("❌ All models failed")
        return None
    
    def generate_personalized_recommendations(
        self, 
        query: str, 
        top_matches: List[Dict[str, Any]], 
        user_role: str = "social_worker"
    ) -> Dict[str, Any]:
        """Generate personalized recommendations using LLM based on query and top matches from database"""
        
        if not self.llm_enabled:
            return self._generate_fallback_recommendations(query, top_matches, user_role)
        
        try:
            # Prepare context from top matches (already retrieved from database)
            context_summary = self._prepare_case_context(top_matches)
            
            # Create comprehensive prompt for LLM
            prompt = self._create_recommendations_prompt(query, context_summary, user_role)
            
            # Try models until we get a valid JSON response
            for model in self.models:
                try:
                    # Get LLM response for this specific model
                    llm_result = self._call_openrouter_single_model(prompt, max_tokens=2500, model=model)
                    
                    if llm_result:
                        llm_response, used_model = llm_result
                        # Try to parse the response
                        try:
                            return self._parse_llm_recommendations(llm_response, query, top_matches, used_model)
                        except (json.JSONDecodeError, ValueError) as parse_error:
                            logger.warning(f"⚠️ Model {model} returned invalid JSON, trying next model: {parse_error}")
                            continue  # Try next model
                    else:
                        logger.warning(f"⚠️ Model {model} failed to respond, trying next model")
                        continue  # Try next model
                        
                except Exception as e:
                    logger.warning(f"⚠️ Model {model} error: {e}")
                    continue  # Try next model
            
            # If we get here, all models failed or returned invalid JSON
            logger.warning("All models failed or returned invalid JSON, using fallback recommendations")
            return self._generate_fallback_recommendations(query, top_matches, user_role)
                
        except Exception as e:
            logger.error(f"Error generating personalized recommendations: {e}")
            return self._generate_fallback_recommendations(query, top_matches, user_role)
    
    def _prepare_case_context(self, top_matches: List[Dict[str, Any]]) -> str:
        """Prepare a summary of the top matching cases for LLM context"""
        context_parts = []
        
        for i, case in enumerate(top_matches[:5], 1):  # Limit to top 5 for context
            case_summary = f"""
Case {i}: {case.get('title', 'Untitled')}
- Risk Types: {', '.join(case.get('risk_types', []))}
- Agencies: {', '.join(case.get('agencies', []))}
- Key Issues: {', '.join(case.get('barriers', [])[:3])}
- Outcome: {case.get('outcome', 'Not specified')}
- Summary: {case.get('summary', '')[:200]}...
"""
            context_parts.append(case_summary)
        
        return "\n".join(context_parts)
    
    def _create_recommendations_prompt(
        self, 
        query: str, 
        context_summary: str, 
        user_role: str
    ) -> str:
        """Create a comprehensive prompt for the LLM"""
        
        role_context = {
            "social_worker": "frontline social worker conducting case assessments and interventions",
            "manager": "social work manager overseeing cases and providing supervision",
            "reviewer": "case review specialist analyzing patterns and outcomes",
            "trainer": "social work trainer developing learning materials"
        }
        
        role_description = role_context.get(user_role, "social work professional")
        
        prompt = f"""
You are an expert {role_description} with extensive experience in child protection and social work. 
You are analyzing case reviews to provide personalized, actionable recommendations.

SEARCH QUERY: "{query}"

TOP MATCHING CASES:
{context_summary}

Based on the search query and the above case examples, provide a comprehensive analysis and recommendations in the following JSON format:

{{
    "query_analysis": {{
        "relevance_summary": "Brief explanation of how the cases relate to the search query",
        "key_patterns": ["pattern1", "pattern2", "pattern3"],
        "risk_factors_identified": ["risk1", "risk2", "risk3"],
        "intervention_opportunities": ["opportunity1", "opportunity2"]
    }},
    "personalized_recommendations": {{
        "immediate_actions": [
            {{
                "action": "Specific action to take",
                "rationale": "Why this action is important",
                "priority": "high/medium/low"
            }}
        ],
        "short_term_strategies": [
            {{
                "strategy": "Strategy description",
                "expected_outcome": "What this should achieve",
                "timeline": "When to implement"
            }}
        ],
        "long_term_approaches": [
            {{
                "approach": "Long-term approach",
                "benefits": "Long-term benefits",
                "considerations": "Things to keep in mind"
            }}
        ]
    }},
    "case_specific_insights": [
        {{
            "case_id": "ID of relevant case",
            "key_lessons": ["lesson1", "lesson2"],
            "applicable_strategies": ["strategy1", "strategy2"],
            "cautionary_notes": ["note1", "note2"]
        }}
    ],
    "professional_development": {{
        "learning_points": ["learning1", "learning2"],
        "skill_development": ["skill1", "skill2"],
        "supervision_topics": ["topic1", "topic2"]
    }},
    "risk_management": {{
        "current_risks": ["risk1", "risk2"],
        "mitigation_strategies": ["strategy1", "strategy2"],
        "escalation_triggers": ["trigger1", "trigger2"]
    }}
}}

Focus on practical, evidence-based recommendations that a {role_description} can implement immediately. 
Consider the specific context of the search query and how the case examples demonstrate successful or unsuccessful interventions.
Ensure all recommendations are specific, actionable, and relevant to the user's role and the search context.
"""
        
        return prompt
    
    def _parse_llm_recommendations(
        self, 
        llm_response: str, 
        query: str, 
        top_matches: List[Dict[str, Any]],
        used_model: str = None
    ) -> Dict[str, Any]:
        """Parse the LLM response into structured recommendations"""
        
        # Extract JSON content from response
        start_idx = llm_response.find('{')
        end_idx = llm_response.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_content = llm_response[start_idx:end_idx]
            parsed_response = json.loads(json_content)
            
            # Add metadata
            parsed_response["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "query": query,
                "cases_analyzed": len(top_matches),
                "llm_model": used_model or self.models[0] if self.models else "unknown",
                "source": "openrouter_llm"
            }
            
            return parsed_response
        else:
            raise ValueError("No JSON content found in LLM response")
    
    def _generate_fallback_recommendations(
        self, 
        query: str, 
        top_matches: List[Dict[str, Any]], 
        user_role: str
    ) -> Dict[str, Any]:
        """Generate fallback recommendations when LLM is not available"""
        
        # Extract common patterns from top matches
        all_risk_types = []
        all_agencies = []
        all_barriers = []
        
        for case in top_matches:
            all_risk_types.extend(case.get('risk_types', []))
            all_agencies.extend(case.get('agencies', []))
            all_barriers.extend(case.get('barriers', []))
        
        # Count frequencies
        risk_type_counts = {}
        for risk_type in all_risk_types:
            risk_type_counts[risk_type] = risk_type_counts.get(risk_type, 0) + 1
        
        agency_counts = {}
        for agency in all_agencies:
            agency_counts[agency] = agency_counts.get(agency, 0) + 1
        
        # Generate recommendations based on patterns
        top_risks = sorted(risk_type_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_agencies = sorted(agency_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "query_analysis": {
                "relevance_summary": f"Analysis based on {len(top_matches)} relevant cases",
                "key_patterns": [risk for risk, count in top_risks],
                "risk_factors_identified": [risk for risk, count in top_risks],
                "intervention_opportunities": [f"Coordinate with {agency}" for agency, count in top_agencies]
            },
            "personalized_recommendations": {
                "immediate_actions": [
                    {
                        "action": f"Assess for {top_risks[0][0] if top_risks else 'identified risks'}",
                        "rationale": "Most common risk factor in similar cases",
                        "priority": "high"
                    },
                    {
                        "action": f"Coordinate with {top_agencies[0][0] if top_agencies else 'relevant agencies'}",
                        "rationale": "Most involved agency in similar cases",
                        "priority": "high"
                    }
                ],
                "short_term_strategies": [
                    {
                        "strategy": "Implement risk assessment based on identified patterns",
                        "expected_outcome": "Early identification of risk factors",
                        "timeline": "Within 1 week"
                    }
                ],
                "long_term_approaches": [
                    {
                        "approach": "Develop multi-agency coordination protocols",
                        "benefits": "Improved intervention effectiveness",
                        "considerations": "Resource allocation and training needs"
                    }
                ]
            },
            "case_specific_insights": [
                {
                    "case_id": case.get('id', 'unknown'),
                    "key_lessons": case.get('barriers', [])[:2],
                    "applicable_strategies": [f"Address {barrier}" for barrier in case.get('barriers', [])[:2]],
                    "cautionary_notes": ["Based on pattern analysis", "Manual review recommended"]
                }
                for case in top_matches[:3]
            ],
            "professional_development": {
                "learning_points": [f"Understanding {risk} patterns" for risk, count in top_risks[:2]],
                "skill_development": ["Risk assessment", "Multi-agency coordination"],
                "supervision_topics": ["Case pattern analysis", "Intervention effectiveness"]
            },
            "risk_management": {
                "current_risks": [risk for risk, count in top_risks[:3]],
                "mitigation_strategies": ["Early intervention", "Regular monitoring", "Multi-agency coordination"],
                "escalation_triggers": ["Multiple risk factors", "Agency coordination failure"]
            },
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "query": query,
                "cases_analyzed": len(top_matches),
                "source": "fallback_pattern_analysis"
            }
        }

# Initialize the service
openrouter_insights_service = OpenRouterInsightsService()

import { useAuthStore } from './auth';

// API base URL - update this to match your backend
const API_BASE_URL = 'https://casereviewer.onrender.com';

export interface SearchFilters {
  riskType?: string[];
  outcome?: string;
  reviewDate?: string;
}

export interface TimelineEvent {
  id: string;
  event_date: string;
  event_type: string;
  description: string;
  impact?: string;
  created_at?: string;
  updated_at?: string;
}

export interface SearchResult {
  id: string;
  title: string;
  summary: string;
  child_age?: number;
  risk_types: string[];
  outcome?: string;
  review_date?: string;
  agencies: string[];
  warning_signs_early: string[];
  risk_factors: string[];
  barriers: string[];
  relationship_model?: {
    familyStructure: string;
    professionalNetwork: string;
    supportSystems: string;
    powerDynamics: string;
  };
  source_file?: string;
  documentUrl?: string;
  similarity_score: number;
  timeline_events: TimelineEvent[];
}

export interface AIRecommendations {
  query_analysis: {
    relevance_summary: string;
    key_patterns: string[];
    risk_factors_identified: string[];
    intervention_opportunities: string[];
  };
  personalized_recommendations: {
    immediate_actions: Array<{
      action: string;
      rationale: string;
      priority: 'high' | 'medium' | 'low';
    }>;
    short_term_strategies: Array<{
      strategy: string;
      expected_outcome: string;
      timeline: string;
    }>;
    long_term_approaches: Array<{
      approach: string;
      benefits: string;
      considerations: string;
    }>;
  };
  case_specific_insights: Array<{
    case_id: string;
    key_lessons: string[];
    applicable_strategies: string[];
    cautionary_notes: string[];
  }>;
  professional_development: {
    learning_points: string[];
    skill_development: string[];
    supervision_topics: string[];
  };
  risk_management: {
    current_risks: string[];
    mitigation_strategies: string[];
    escalation_triggers: string[];
  };
  metadata: {
    generated_at: string;
    query: string;
    cases_analyzed: number;
    llm_model: string;
    source: string;
  };
}

export interface SearchResponse {
  results: SearchResult[];
  totalCount: number;
  searchTime: number;
  ai_recommendations?: AIRecommendations;
  message?: string;
}

const getAuthHeaders = () => {
  const token = useAuthStore.getState().token;
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  };
};

export const searchCaseReviews = async (
  query: string,
  filters?: SearchFilters
): Promise<SearchResponse> => {
  const response = await fetch(`${API_BASE_URL}/api/protected/search`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({ 
      query, 
      filters,
      top_k: 10 // Request top 10 results for AI analysis
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Search failed');
  }

  const data = await response.json();
  
  // Log AI recommendations for debugging
  if (data.ai_recommendations) {
    console.log('AI Recommendations received:', data.ai_recommendations);
  }
  
  return data;
};

export const getCaseReview = async (id: string): Promise<SearchResult> => {
  const response = await fetch(`${API_BASE_URL}/api/protected/case-reviews/${id}`, {
    headers: getAuthHeaders(),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to fetch case review');
  }

  return response.json();
};

export const getSearchHistory = async () => {
  const response = await fetch(`${API_BASE_URL}/api/protected/search-history`, {
    headers: getAuthHeaders(),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to fetch search history');
  }

  return response.json();
};

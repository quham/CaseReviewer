import { useAuthStore } from './auth';

export interface SearchFilters {
  childAge?: string;
  riskType?: string;
  outcome?: string;
  reviewDate?: string;
}

export interface TimelineEvent {
  id: string;
  eventDate: string;
  eventType: string;
  description: string;
  outcome?: string;
  details?: string;
  track: string;
}

export interface SearchResult {
  id: string;
  title: string;
  summary: string;
  details: string;
  childAge?: number; // should this be a range
  riskTypes: string[];
  outcome?: string;
  reviewDate: string;
  agencies: string[];
  warningSignsEarly: string[];
  riskFactors: string[];
  barriers: string[];
  relationshipModel?: {
    familyStructure: string;
    professionalNetwork: string;
    supportSystems: string;
    powerDynamics: string;
  };
  documentUrl?: string;
  relevanceScore: number;
  keyMatches: string[];
  aiAdvice?: string;
  timelineEvents: TimelineEvent[];
}

export interface SearchResponse {
  results: SearchResult[];
  totalCount: number;
  searchTime: number;
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
  const response = await fetch('/api/protected/search', {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({ query, filters }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Search failed');
  }

  return response.json();
};

export const getCaseReview = async (id: string): Promise<SearchResult> => {
  const response = await fetch(`/api/protected/case-reviews/${id}`, {
    headers: getAuthHeaders(),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to fetch case review');
  }

  return response.json();
};

export const getSearchHistory = async () => {
  const response = await fetch('/api/protected/search-history', {
    headers: getAuthHeaders(),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to fetch search history');
  }

  return response.json();
};

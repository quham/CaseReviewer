import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Card, CardContent } from '@/components/ui/card';
import { MultiSelect, type MultiSelectOption } from '@/components/ui/multi-select';
import { Search, History, Info } from 'lucide-react';
import { searchCaseReviews, type SearchFilters, type SearchResult } from '@/lib/search-api';
import { useToast } from '@/hooks/use-toast';
import { usePreferencesStore } from '@/lib/preferences';

interface SearchInterfaceProps {
  onResults: (results: SearchResult[], searchTime: number, totalCount: number, aiRecommendations?: any, message?: string, filters?: SearchFilters) => void;
  onSearchStart: () => void;
  onSearchEnd: () => void;
}

export function SearchInterface({ onResults, onSearchStart, onSearchEnd }: SearchInterfaceProps) {
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState<SearchFilters>({});
  const { toast } = useToast();
  const { searchHistoryEnabled } = usePreferencesStore();

  const caseTypeOptions: MultiSelectOption[] = [
    { value: 'child_abuse', label: 'Child Abuse' },
    { value: 'neglect', label: 'Neglect' },
    { value: 'domestic_violence', label: 'Domestic Violence' },
    { value: 'sexual_abuse', label: 'Sexual Abuse' },
    { value: 'emotional_abuse', label: 'Emotional Abuse' },
    { value: 'other', label: 'Other' }
  ];

  // Function to sort results by risk type matches
  const sortResultsByRiskType = (results: SearchResult[], selectedRiskTypes?: string[]): SearchResult[] => {
    if (!selectedRiskTypes || selectedRiskTypes.length === 0) {
      return results; // No sorting needed if no risk types selected
    }

    return [...results].sort((a, b) => {
      const aMatches = a.risk_types?.some(type => selectedRiskTypes.includes(type)) || false;
      const bMatches = b.risk_types?.some(type => selectedRiskTypes.includes(type)) || false;
      
      if (aMatches && !bMatches) return -1; // a comes first
      if (!aMatches && bMatches) return 1;  // b comes first
      return 0; // both match or both don't match, maintain original order
    });
  };

  const handleSearch = async () => {
    if (!query.trim()) {
      toast({
        title: "Search Required",
        description: "Please enter a case description to search.",
        variant: "destructive",
      });
      return;
    }

    onSearchStart();
    
    try {
      const response = await searchCaseReviews(query, filters);
      const sortedResults = sortResultsByRiskType(response.results, filters.riskType);
      
      // Pass the selected risk types along with the results for highlighting
      const resultsWithFilters = {
        results: sortedResults,
        selectedRiskTypes: filters.riskType || [],
        searchTime: response.searchTime,
        totalCount: response.totalCount,
        aiRecommendations: response.ai_recommendations,
        message: response.message
      };
      
      onResults(sortedResults, response.searchTime, response.totalCount, response.ai_recommendations, response.message, filters);
      
      // Show a toast based on whether we got results or just general guidance
      if (sortedResults.length > 0) {
        const aiMessage = response.ai_recommendations ? ' with AI-powered insights' : '';
        const riskTypeMessage = filters.riskType && filters.riskType.length > 0 
          ? ` (${filters.riskType.length} risk type${filters.riskType.length > 1 ? 's' : ''} selected)`
          : '';
        toast({
          title: "Search Completed",
          description: `Found ${response.totalCount} relevant case reviews${response.searchTime > 0 ? ` in ${(response.searchTime / 1000).toFixed(1)} seconds` : ''}${aiMessage}${riskTypeMessage}`,
        });
      } else if (response.message) {
        toast({
          title: "Professional Guidance Available",
          description: "No case matches found, but general professional guidance has been provided below.",
        });
      }
    } catch (error) {
      toast({
        title: "Search Failed",
        description: error instanceof Error ? error.message : "Failed to perform search",
        variant: "destructive",
      });
    } finally {
      onSearchEnd();
    }
  };

  const characterCount = query.length;
  const maxCharacters = 1000;

  return (
    <Card className="mb-8">
      <CardContent className="pt-6">
        <div className="space-y-6">
          <div>
            <Label htmlFor="search-query" className="text-base font-medium text-gray-900 mb-3 block">
              Describe the case you're working on
            </Label>
            <Textarea
              id="search-query"
              placeholder="e.g., A 7-year-old child with unexplained bruising, parents give inconsistent explanations, child appears withdrawn and anxious around father..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="min-h-[120px] resize-none"
              maxLength={maxCharacters}
            />
            <div className="flex justify-between items-center mt-2">
              <span className="text-sm text-gray-500">
                Be as detailed as possible for better case matches
              </span>
              <span className={`text-sm ${characterCount > maxCharacters * 0.9 ? 'text-orange-600' : 'text-gray-500'}`}>
                {characterCount}/{maxCharacters}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label className="block text-sm font-medium text-neutral-700 mb-1">
                Case Type
              </Label>
    
              <MultiSelect
                options={caseTypeOptions}
                selected={filters.riskType || []}
                onChange={(selected) => setFilters({ ...filters, riskType: selected.length > 0 ? selected : undefined })}
                placeholder="Select case types..."
                className="w-full"
              />
              <p className="text-xs text-neutral-500 mb-2">Select one or more case types (leave empty for all types)</p>
            </div>
            
            <div>
              <Label className="block text-sm font-medium text-neutral-700 mb-1">Review Date</Label>
              <Select value={filters.reviewDate || 'any'} onValueChange={(value) => setFilters({ ...filters, reviewDate: value === 'any' ? undefined : value })}>
                <SelectTrigger>
                  <SelectValue placeholder="Any date" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="any">Any date</SelectItem>
                  <SelectItem value="last_2_years">Last 2 years</SelectItem>
                  <SelectItem value="last_5_years">Last 5 years</SelectItem>
                  <SelectItem value="last_10_years">Last 10 years</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button
                className="bg-secondary hover:bg-blue-600 text-white px-6 py-3 rounded-lg font-medium transition-colors flex items-center"
                onClick={handleSearch}
                disabled={!query.trim()}
                data-testid="button-search"
              >
                <Search className="w-4 h-4 mr-2" />
                Search Case Reviews
              </Button>
              
              {searchHistoryEnabled && (
                <Button 
                  variant="ghost"
                  className="text-neutral-600 hover:text-neutral-800 px-4 py-3 rounded-lg transition-colors"
                  data-testid="button-search-history"
                >
                  <History className="w-4 h-4 mr-2" />
                  Search History
                </Button>
              )}
            </div>
            
            <div className="text-sm text-neutral-500 flex items-center">
              <Info className="w-4 h-4 mr-1" />
              {searchHistoryEnabled 
                ? "Search history is enabled for quick access to recent searches"
                : "Search queries are not stored for privacy protection"
              }
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

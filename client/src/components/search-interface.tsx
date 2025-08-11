import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Card, CardContent } from '@/components/ui/card';
import { Search, History, Info } from 'lucide-react';
import { searchCaseReviews, type SearchFilters, type SearchResult } from '@/lib/search-api';
import { useToast } from '@/hooks/use-toast';

interface SearchInterfaceProps {
  onResults: (results: SearchResult[], searchTime: number, totalCount: number) => void;
  onSearchStart: () => void;
  onSearchEnd: () => void;
}

export function SearchInterface({ onResults, onSearchStart, onSearchEnd }: SearchInterfaceProps) {
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState<SearchFilters>({});
  const { toast } = useToast();

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
      onResults(response.results, response.searchTime, response.totalCount);
      
      toast({
        title: "Search Completed",
        description: `Found ${response.totalCount} relevant case reviews in ${(response.searchTime / 1000).toFixed(1)} seconds`,
      });
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
      <CardContent className="p-6">
        <div className="mb-4">
          <Label htmlFor="case-search" className="block text-sm font-medium text-neutral-700 mb-2">
            Describe your current case situation
          </Label>
          <div className="relative">
            <Textarea
              id="case-search"
              rows={4}
              className="w-full px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-secondary focus:border-transparent resize-none"
              placeholder="Example: Emma, age 6, living with mother (24) and mother's new partner (31) of 8 months. Historical DV with Emma's father who has supervised contact monthly. Mother reports new partner 'loses temper' with Emma, particularly around bedtime routines..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              maxLength={maxCharacters}
              data-testid="textarea-case-search"
            />
            <div className="absolute bottom-3 right-3 text-xs text-neutral-400">
              <span data-testid="text-character-count">{characterCount}</span>/{maxCharacters} characters
            </div>
          </div>
        </div>
        
        {/* Search Filters */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
          <div>
            <Label className="block text-sm font-medium text-neutral-700 mb-1">Child Age Range</Label>
            <Select value={filters.childAge || 'Any age'} onValueChange={(value) => setFilters({...filters, childAge: value})}>
              <SelectTrigger data-testid="select-child-age">
                <SelectValue placeholder="Any age" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Any age">Any age</SelectItem>
                <SelectItem value="0-5 years">0-5 years</SelectItem>
                <SelectItem value="6-11 years">6-11 years</SelectItem>
                <SelectItem value="12-17 years">12-17 years</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div>
            <Label className="block text-sm font-medium text-neutral-700 mb-1">Risk Type</Label>
            <Select value={filters.riskType || 'All types'} onValueChange={(value) => setFilters({...filters, riskType: value})}>
              <SelectTrigger data-testid="select-risk-type">
                <SelectValue placeholder="All types" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="All types">All types</SelectItem>
                <SelectItem value="Domestic violence">Domestic violence</SelectItem>
                <SelectItem value="Neglect">Neglect</SelectItem>
                <SelectItem value="Physical abuse">Physical abuse</SelectItem>
                <SelectItem value="Emotional abuse">Emotional abuse</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div>
            <Label className="block text-sm font-medium text-neutral-700 mb-1">Case Outcome</Label>
            <Select value={filters.outcome || 'All outcomes'} onValueChange={(value) => setFilters({...filters, outcome: value})}>
              <SelectTrigger data-testid="select-outcome">
                <SelectValue placeholder="All outcomes" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="All outcomes">All outcomes</SelectItem>
                <SelectItem value="Successful intervention">Successful intervention</SelectItem>
                <SelectItem value="Ongoing support">Ongoing support</SelectItem>
                <SelectItem value="Case closure">Case closure</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div>
            <Label className="block text-sm font-medium text-neutral-700 mb-1">Review Date</Label>
            <Select value={filters.reviewDate || 'Any date'} onValueChange={(value) => setFilters({...filters, reviewDate: value})}>
              <SelectTrigger data-testid="select-review-date">
                <SelectValue placeholder="Any date" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Any date">Any date</SelectItem>
                <SelectItem value="Last 2 years">Last 2 years</SelectItem>
                <SelectItem value="Last 5 years">Last 5 years</SelectItem>
                <SelectItem value="Last 10 years">Last 10 years</SelectItem>
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
            
            <Button 
              variant="ghost"
              className="text-neutral-600 hover:text-neutral-800 px-4 py-3 rounded-lg transition-colors"
              data-testid="button-search-history"
            >
              <History className="w-4 h-4 mr-2" />
              Search History
            </Button>
          </div>
          
          <div className="text-sm text-neutral-500 flex items-center">
            <Info className="w-4 h-4 mr-1" />
            Search queries are not stored for privacy protection
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

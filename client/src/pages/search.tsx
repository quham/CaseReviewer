import { useState } from 'react';
import { Header } from '@/components/header';
import { SearchInterface } from '@/components/search-interface';
import { SearchResults } from '@/components/search-results';
import type { SearchResult, SearchFilters } from '@/lib/search-api';

export default function SearchPage() {
  const [searchResults, setSearchResults] = useState<SearchResult[] | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [searchTime, setSearchTime] = useState<number>(0);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [aiRecommendations, setAiRecommendations] = useState<any>(null);
  const [message, setMessage] = useState<string>('');
  const [selectedRiskTypes, setSelectedRiskTypes] = useState<string[]>([]);

  const handleSearchResults = (results: SearchResult[], time: number, count: number, aiRecs?: any, msg?: string, filters?: SearchFilters) => {
    setSearchResults(results);
    setSearchTime(time);
    setTotalCount(count);
    setAiRecommendations(aiRecs);
    setMessage(msg || '');
    setSelectedRiskTypes(filters?.riskType || []);
  };

  return (
    <div className="min-h-screen bg-neutral-50">
      <Header />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <SearchInterface 
          onResults={handleSearchResults}
          onSearchStart={() => setIsSearching(true)}
          onSearchEnd={() => setIsSearching(false)}
        />

        {(searchResults !== null || isSearching) && (
          <SearchResults
            results={searchResults || []}
            isLoading={isSearching}
            searchTime={searchTime}
            totalCount={totalCount}
            aiRecommendations={aiRecommendations}
            message={message}
            selectedRiskTypes={selectedRiskTypes}
          />
        )}
      </main>

      <footer className="bg-primary text-white mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center">
            <h3 className="font-semibold mb-4">NSPCC Case Review Search</h3>
            <p className="text-neutral-300 text-sm mb-6">
              Supporting social work professionals with child-focused, evidence-based insights and recommendations from historical case reviews. 
            </p>
            <div className="border-t border-neutral-700 pt-6 text-sm text-neutral-400">
              <p>&copy; 2025. All data is handled in accordance with GDPR and professional standards.</p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

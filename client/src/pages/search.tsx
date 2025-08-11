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

  const handleSearchResults = (results: SearchResult[], time: number, count: number) => {
    setSearchResults(results);
    setSearchTime(time);
    setTotalCount(count);
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
          />
        )}
      </main>

      <footer className="bg-primary text-white mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="font-semibold mb-4">NSPCC Case Review Search</h3>
              <p className="text-neutral-300 text-sm">
                Supporting social work professionals with evidence-based insights from historical case reviews.
              </p>
            </div>
            <div>
              <h3 className="font-semibold mb-4">Professional Resources</h3>
              <ul className="space-y-2 text-sm text-neutral-300">
                <li><a href="#" className="hover:text-white transition-colors">Professional Guidelines</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Training Materials</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Support Contacts</a></li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-4">System Information</h3>
              <ul className="space-y-2 text-sm text-neutral-300">
                <li><a href="#" className="hover:text-white transition-colors">Privacy Policy</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Data Protection</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Technical Support</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-neutral-700 mt-8 pt-8 text-center text-sm text-neutral-400">
            <p>&copy; 2024 NSPCC. This system is for authorized social work professionals only. All data is handled in accordance with GDPR and professional standards.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

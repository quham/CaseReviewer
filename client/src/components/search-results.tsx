import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Download, Printer } from 'lucide-react';
import { ResultCard } from './result-card';
import type { SearchResult } from '@/lib/search-api';

interface SearchResultsProps {
  results: SearchResult[];
  isLoading: boolean;
  searchTime: number;
  totalCount: number;
}

export function SearchResults({ results, isLoading, searchTime, totalCount }: SearchResultsProps) {
  if (isLoading) {
    return (
      <div className="space-y-8">
        <Card>
          <CardContent className="p-4">
            <Skeleton className="h-4 w-48 mb-2" />
            <Skeleton className="h-4 w-96" />
          </CardContent>
        </Card>
        
        {Array.from({ length: 3 }).map((_, index) => (
          <Card key={index} className="overflow-hidden">
            <div className="bg-gradient-to-r from-secondary to-blue-600 p-6">
              <Skeleton className="h-6 w-64 bg-white/20 mb-2" />
              <Skeleton className="h-4 w-full bg-white/10" />
            </div>
            <CardContent className="p-6">
              <Skeleton className="h-32 w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (!results.length) {
    return null;
  }

  return (
    <div className="space-y-8" id="search-results">
      {/* Results Header */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-primary" data-testid="text-results-title">
                Search Results
              </h2>
              <p className="text-sm text-neutral-600 mt-1" data-testid="text-results-summary">
                Found <span className="font-medium">{totalCount}</span> relevant case reviews in{' '}
                <span className="font-medium">{(searchTime / 1000).toFixed(1)}</span> seconds
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <Button
                variant="ghost"
                size="sm"
                className="text-neutral-600 hover:text-neutral-800"
                data-testid="button-export-results"
              >
                <Download className="w-4 h-4 mr-2" />
                Export Results
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="text-neutral-600 hover:text-neutral-800"
                data-testid="button-print-results"
              >
                <Printer className="w-4 h-4 mr-2" />
                Print
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Case Review Results */}
      {results.map((result, index) => (
        <ResultCard
          key={result.id}
          result={result}
          isExpanded={index === 0} // First result is expanded by default
        />
      ))}

      {/* Load More Results */}
      {results.length < totalCount && (
        <div className="text-center mt-8">
          <Button
            variant="outline"
            className="px-6 py-3"
            data-testid="button-load-more"
          >
            Load More Results
          </Button>
        </div>
      )}
    </div>
  );
}

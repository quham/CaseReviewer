import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Bookmark, 
  Copy, 
  Download, 
  ExternalLink, 
  ChevronDown, 
  ChevronUp,
  Bot,
  Quote,
  Lightbulb,
  Clock,
  FileText,
  Users,
  AlertTriangle,
  Shield,
  Zap
} from 'lucide-react';
import { CaseTimeline } from './case-timeline';
import type { SearchResult } from '@/lib/search-api';
import { useToast } from '@/hooks/use-toast';

interface ResultCardProps {
  result: SearchResult;
  isExpanded?: boolean;
}

export function ResultCard({ result, isExpanded = false }: ResultCardProps) {
  const [expanded, setExpanded] = useState(isExpanded);
  const { toast } = useToast();

  const getOutcomeBadgeColor = (outcome?: string) => {
    switch (outcome) {
      case 'Successful intervention':
        return 'bg-success text-white';
      case 'Ongoing support':
        return 'bg-warning text-white';
      case 'Case closure':
        return 'bg-neutral-600 text-white';
      default:
        return 'bg-neutral-500 text-white';
    }
  };

  const copyToClipboard = async (text: string, type: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast({
        title: "Copied to Clipboard",
        description: `${type} has been copied to your clipboard.`,
      });
    } catch (error) {
      toast({
        title: "Copy Failed",
        description: "Unable to copy to clipboard. Please try again.",
        variant: "destructive",
      });
    }
  };

  if (!expanded) {
    return (
      <Card className="overflow-hidden">
        <div className="bg-gradient-to-r from-neutral-100 to-neutral-200 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Badge className="bg-neutral-600 text-white">
                {result.relevanceScore}% Match
              </Badge>
              <h3 className="font-semibold text-primary" data-testid={`text-title-${result.id}`}>
                {result.title}
              </h3>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setExpanded(true)}
              data-testid={`button-expand-${result.id}`}
            >
              <ChevronDown className="w-4 h-4" />
            </Button>
          </div>
          <p className="text-sm text-neutral-600 mt-2" data-testid={`text-summary-${result.id}`}>
            {result.summary.substring(0, 150)}...
          </p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="overflow-hidden">
      {/* Key Information Header */}
      <div className="bg-gradient-to-r from-secondary to-blue-600 text-white p-6">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center space-x-2 mb-2">
              <Badge className="bg-white/20 text-white">
                {result.relevanceScore}% Match
              </Badge>
              {result.outcome && (
                <Badge className={getOutcomeBadgeColor(result.outcome)}>
                  <Shield className="w-3 h-3 mr-1" />
                  {result.outcome}
                </Badge>
              )}
            </div>
            <h3 className="text-xl font-semibold mb-2" data-testid={`text-expanded-title-${result.id}`}>
              {result.title}
            </h3>
            <p className="text-blue-100" data-testid={`text-expanded-summary-${result.id}`}>
              {result.summary}
            </p>
          </div>
          <div className="flex space-x-2">
            <Button 
              variant="ghost" 
              size="sm" 
              className="text-white hover:text-blue-200"
              data-testid={`button-bookmark-${result.id}`}
            >
              <Bookmark className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setExpanded(false)}
              className="text-white hover:text-blue-200"
              data-testid={`button-collapse-${result.id}`}
            >
              <ChevronUp className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      <CardContent className="p-6">
        {/* Key Matches */}
        {result.keyMatches.length > 0 && (
          <div className="mb-6">
            <div className="flex items-center space-x-2 mb-3">
              <Bot className="w-4 h-4 text-secondary" />
              <h4 className="font-semibold text-primary">Key Matches</h4>
              <Badge variant="secondary" className="text-xs">AI Generated</Badge>
            </div>
            <div className="bg-blue-50 border-l-4 border-secondary p-4 rounded-r-lg">
              <ul className="text-sm text-neutral-700 space-y-1 list-disc list-inside">
                {result.keyMatches.map((match, index) => (
                  <li key={index} data-testid={`text-key-match-${index}`}>{match}</li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {/* AI-Generated Additional Advice */}
        {result.aiAdvice && (
          <div className="mb-6">
            <div className="flex items-center space-x-2 mb-3">
              <Lightbulb className="w-4 h-4 text-warning" />
              <h4 className="font-semibold text-primary">AI-Generated Additional Advice</h4>
              <Badge variant="outline" className="bg-yellow-100 text-yellow-800 text-xs">AI Generated</Badge>
            </div>
            <div className="bg-yellow-50 border-l-4 border-warning p-4 rounded-r-lg">
              <div className="text-sm text-neutral-700 whitespace-pre-wrap" data-testid={`text-ai-advice-${result.id}`}>
                {result.aiAdvice}
              </div>
            </div>
            <Alert className="mt-3 bg-red-50 border-danger">
              <AlertTriangle className="w-4 h-4" />
              <AlertDescription className="text-xs text-red-700">
                <strong>Professional Responsibility Notice:</strong> This AI-generated advice supplements professional judgment and should not replace comprehensive assessment and multi-agency consultation.
              </AlertDescription>
            </Alert>
          </div>
        )}

        {/* Interactive Case Timeline */}
        {result.timelineEvents && result.timelineEvents.length > 0 && (
          <div className="mb-6">
            <div className="flex items-center justify-between mb-4">
              <h4 className="font-semibold text-primary flex items-center">
                <Clock className="w-4 h-4 text-neutral-400 mr-2" />
                Interactive Case Timeline
              </h4>
            </div>
            <CaseTimeline events={result.timelineEvents} />
          </div>
        )}

        {/* Additional Information Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Case Review Summary */}
          <div className="bg-neutral-50 rounded-lg p-4 border border-neutral-200">
            <h4 className="font-semibold text-primary mb-3 flex items-center">
              <FileText className="w-4 h-4 text-neutral-400 mr-2" />
              Case Review Summary
            </h4>
            <div className="text-sm space-y-2">
              <p><strong>Child Age:</strong> {result.childAge ? `${result.childAge} years` : 'Not specified'}</p>
              <p><strong>Key Agencies:</strong> {result.agencies.join(', ') || 'Not specified'}</p>
              <p><strong>Primary Outcome:</strong> {result.outcome || 'Not specified'}</p>
              <p><strong>Review Date:</strong> {new Date(result.reviewDate).toLocaleDateString()}</p>
            </div>
          </div>

          {/* Relationship Model */}
          {result.relationshipModel && (
            <div className="bg-neutral-50 rounded-lg p-4 border border-neutral-200">
              <h4 className="font-semibold text-primary mb-3 flex items-center">
                <Users className="w-4 h-4 text-neutral-400 mr-2" />
                Relationship Model
              </h4>
              <div className="text-sm space-y-2">
                <p><strong>Family Structure:</strong> {result.relationshipModel.familyStructure}</p>
                <p><strong>Professional Network:</strong> {result.relationshipModel.professionalNetwork}</p>
                <p><strong>Support Systems:</strong> {result.relationshipModel.supportSystems}</p>
                <p><strong>Power Dynamics:</strong> {result.relationshipModel.powerDynamics}</p>
              </div>
            </div>
          )}
        </div>

        {/* Warning Signs, Risk Factors, Barriers */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <div className="bg-red-50 rounded-lg p-4 border border-red-200">
            <h4 className="font-semibold text-danger mb-3 flex items-center">
              <AlertTriangle className="w-4 h-4 mr-2" />
              Warning Signs
            </h4>
            <ul className="text-sm text-red-700 space-y-1 list-disc list-inside">
              {result.warningSignsEarly.map((sign, index) => (
                <li key={index}>{sign}</li>
              ))}
            </ul>
          </div>

          <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
            <h4 className="font-semibold text-warning mb-3 flex items-center">
              <Shield className="w-4 h-4 mr-2" />
              Risk Factors
            </h4>
            <ul className="text-sm text-orange-700 space-y-1 list-disc list-inside">
              {result.riskFactors.map((factor, index) => (
                <li key={index}>{factor}</li>
              ))}
            </ul>
          </div>

          <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
            <h4 className="font-semibold text-yellow-700 mb-3 flex items-center">
              <Zap className="w-4 h-4 mr-2" />
              Identified Barriers
            </h4>
            <ul className="text-sm text-yellow-700 space-y-1 list-disc list-inside">
              {result.barriers.map((barrier, index) => (
                <li key={index}>{barrier}</li>
              ))}
            </ul>
          </div>
        </div>

        {/* Direct Link and Actions */}
        <div className="flex items-center justify-between pt-4 border-t border-neutral-200">
          <div className="flex items-center space-x-4">
            {result.documentUrl && (
              <a 
                href={result.documentUrl}
                className="text-secondary hover:text-blue-600 text-sm transition-colors flex items-center"
                target="_blank"
                rel="noopener noreferrer"
                data-testid={`link-document-${result.id}`}
              >
                <ExternalLink className="w-4 h-4 mr-2" />
                View Full Case Review
              </a>
            )}
            <span className="text-neutral-300">|</span>
            <Button 
              variant="ghost" 
              size="sm" 
              className="text-secondary hover:text-blue-600"
              data-testid={`button-bookmark-case-${result.id}`}
            >
              <Bookmark className="w-4 h-4 mr-1" />
              Bookmark for Case File
            </Button>
          </div>
          
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => copyToClipboard(result.summary, 'Case summary')}
              data-testid={`button-copy-summary-${result.id}`}
            >
              <Copy className="w-4 h-4 mr-2" />
              Copy Summary
            </Button>
            <Button
              className="bg-secondary hover:bg-blue-600 text-white"
              size="sm"
              data-testid={`button-export-case-${result.id}`}
            >
              <Download className="w-4 h-4 mr-2" />
              Export to Case File
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

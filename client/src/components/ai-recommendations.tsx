import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AlertCircle, Lightbulb, Target, TrendingUp, Shield, BookOpen } from 'lucide-react';
import type { AIRecommendations } from '@/lib/search-api';

interface AIRecommendationsProps {
  recommendations: AIRecommendations;
}

export function AIRecommendations({ recommendations }: AIRecommendationsProps) {
  // Add null checks and fallbacks for all properties
  const { 
    query_analysis = {}, 
    personalized_recommendations = {}, 
    case_specific_insights = [], 
    professional_development = {}, 
    risk_management = {},
    metadata = {}
  } = recommendations;

  // Early return if no recommendations are available
  if (!recommendations || Object.keys(recommendations).length === 0) {
    return (
      <div className="space-y-6 mb-8">
        <Card className="border-gray-200 bg-gray-50/50">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center text-gray-800">
              <Lightbulb className="w-5 h-5 mr-2" />
              AI Recommendations
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-600">No AI recommendations available at this time.</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6 mb-8">
      {/* Query Analysis */}
      {query_analysis && query_analysis.relevance_summary && (
        <Card className="border-blue-200 bg-blue-50/50">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center text-blue-800">
              <Lightbulb className="w-5 h-5 mr-2" />
              AI Analysis of Your Search
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-blue-700">{query_analysis.relevance_summary}</p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {query_analysis.key_patterns && query_analysis.key_patterns.length > 0 && (
                <div>
                  <h4 className="font-semibold text-blue-800 mb-2">Key Patterns Identified</h4>
                  <div className="flex flex-wrap gap-2">
                    {query_analysis.key_patterns.map((pattern, index) => (
                      <Badge key={index} variant="secondary" className="bg-blue-100 text-blue-800">
                        {pattern}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
              
              {query_analysis.risk_factors_identified && query_analysis.risk_factors_identified.length > 0 && (
                <div>
                  <h4 className="font-semibold text-blue-800 mb-2">Risk Factors</h4>
                  <div className="flex flex-wrap gap-2">
                    {query_analysis.risk_factors_identified.map((risk, index) => (
                      <Badge key={index} variant="destructive" className="bg-red-100 text-red-800">
                        {risk}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Personalized Recommendations */}
      {personalized_recommendations && Object.keys(personalized_recommendations).length > 0 && (
        <Card className="border-green-200 bg-green-50/50">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center text-green-800">
              <Target className="w-5 h-5 mr-2" />
              Personalized Recommendations
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Immediate Actions */}
            {personalized_recommendations.immediate_actions && personalized_recommendations.immediate_actions.length > 0 && (
              <div>
                <h4 className="font-semibold text-green-800 mb-3 flex items-center">
                  <AlertCircle className="w-4 h-4 mr-2" />
                  Immediate Actions
                </h4>
                <div className="space-y-3">
                  {personalized_recommendations.immediate_actions.map((action, index) => (
                    <div key={index} className="bg-white p-3 rounded-lg border border-green-200">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <p className="font-medium text-gray-900">{action.action}</p>
                          <p className="text-sm text-gray-600 mt-1">{action.rationale}</p>
                        </div>
                        <Badge 
                          variant={action.priority === 'high' ? 'destructive' : action.priority === 'medium' ? 'default' : 'secondary'}
                          className="ml-3"
                        >
                          {action.priority}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Short-term Strategies */}
            {personalized_recommendations.short_term_strategies && personalized_recommendations.short_term_strategies.length > 0 && (
              <div>
                <h4 className="font-semibold text-green-800 mb-3 flex items-center">
                  <TrendingUp className="w-4 h-4 mr-2" />
                  Short-term Strategies
                </h4>
                <div className="space-y-3">
                  {personalized_recommendations.short_term_strategies.map((strategy, index) => (
                    <div key={index} className="bg-white p-3 rounded-lg border border-green-200">
                      <p className="font-medium text-gray-900">{strategy.strategy}</p>
                      <p className="text-sm text-gray-600 mt-1">
                        <span className="font-medium">Expected Outcome:</span> {strategy.expected_outcome}
                      </p>
                      <p className="text-sm text-gray-600">
                        <span className="font-medium">Timeline:</span> {strategy.timeline}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Long-term Approaches */}
            {personalized_recommendations.long_term_approaches && personalized_recommendations.long_term_approaches.length > 0 && (
              <div>
                <h4 className="font-semibold text-green-800 mb-3 flex items-center">
                  <Target className="w-4 h-4 mr-2" />
                  Long-term Approaches
                </h4>
                <div className="space-y-3">
                  {personalized_recommendations.long_term_approaches.map((approach, index) => (
                    <div key={index} className="bg-white p-3 rounded-lg border border-green-200">
                      <p className="font-medium text-gray-900">{approach.approach}</p>
                      <p className="text-sm text-gray-600 mt-1">
                        <span className="font-medium">Benefits:</span> {approach.benefits}
                      </p>
                      <p className="text-sm text-gray-600">
                        <span className="font-medium">Considerations:</span> {approach.considerations}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Professional Development */}
      {professional_development && Object.keys(professional_development).length > 0 && (
        <Card className="border-purple-200 bg-purple-50/50">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center text-purple-800">
              <BookOpen className="w-5 h-5 mr-2" />
              Professional Development
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {professional_development.learning_points && professional_development.learning_points.length > 0 && (
                <div>
                  <h4 className="font-semibold text-purple-800 mb-2">Learning Points</h4>
                  <ul className="space-y-2">
                    {professional_development.learning_points.map((point, index) => (
                      <li key={index} className="flex items-start">
                        <div className="w-2 h-2 bg-purple-400 rounded-full mt-2 mr-3 flex-shrink-0" />
                        <span className="text-sm text-gray-700">{point}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              
              {professional_development.skill_development && professional_development.skill_development.length > 0 && (
                <div>
                  <h4 className="font-semibold text-purple-800 mb-2">Skill Development</h4>
                  <ul className="space-y-2">
                    {professional_development.skill_development.map((skill, index) => (
                      <li key={index} className="flex items-start">
                        <div className="w-2 h-2 bg-purple-400 rounded-full mt-2 mr-3 flex-shrink-0" />
                        <span className="text-sm text-gray-700">{skill}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Risk Management */}
      {risk_management && Object.keys(risk_management).length > 0 && (
        <Card className="border-orange-200 bg-orange-50/50">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center text-orange-800">
              <Shield className="w-5 h-5 mr-2" />
              Risk Management
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {risk_management.current_risks && risk_management.current_risks.length > 0 && (
                <div>
                  <h4 className="font-semibold text-orange-800 mb-2">Current Risks</h4>
                  <div className="flex flex-wrap gap-2">
                    {risk_management.current_risks.map((risk, index) => (
                      <Badge key={index} variant="destructive" className="bg-red-100 text-red-800 text-xs">
                        {risk}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
              
              {risk_management.mitigation_strategies && risk_management.mitigation_strategies.length > 0 && (
                <div>
                  <h4 className="font-semibold text-orange-800 mb-2">Mitigation Strategies</h4>
                  <ul className="space-y-1">
                    {risk_management.mitigation_strategies.map((strategy, index) => (
                      <li key={index} className="text-sm text-gray-700">
                        • {strategy}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              
              {risk_management.escalation_triggers && risk_management.escalation_triggers.length > 0 && (
                <div>
                  <h4 className="font-semibold text-orange-800 mb-2">Escalation Triggers</h4>
                  <ul className="space-y-1">
                    {risk_management.escalation_triggers.map((trigger, index) => (
                      <li key={index} className="text-sm text-gray-700">
                        • {trigger}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Metadata */}
      {metadata && Object.keys(metadata).length > 0 && (
        <div className="text-center text-xs text-gray-500">
          {metadata.cases_analyzed && <p>Based on analysis of {metadata.cases_analyzed} relevant cases</p>}
          {metadata.generated_at && <p>Generated at {new Date(metadata.generated_at).toLocaleString()}</p>}
        </div>
      )}
    </div>
  );
}

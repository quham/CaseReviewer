import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { ChevronDown, Clock, Calendar, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';
import type { TimelineEvent } from '@/lib/search-api';

interface CaseTimelineProps {
  events: TimelineEvent[];
}

export function CaseTimeline({ events }: CaseTimelineProps) {
  const [showDetails, setShowDetails] = useState(false);

  // Validate and filter events
  const validEvents = events?.filter(event => {
    if (!event || !event.id || !event.event_date || !event.description) {
      return false;
    }
    
    // Validate date format
    const date = new Date(event.event_date);
    if (isNaN(date.getTime())) {
      return false;
    }
    
    return true;
  }) || [];

  // Get date range
  const sortedEvents = [...validEvents].sort((a, b) => {
    try {
      return new Date(a.event_date).getTime() - new Date(b.event_date).getTime();
    } catch (error) {
      return 0;
    }
  });

  const startDate = sortedEvents[0]?.event_date;
  const endDate = sortedEvents[sortedEvents.length - 1]?.event_date;

  const getOutcomeColor = (outcome?: string) => {
    if (!outcome) return 'bg-slate-400';
    
    switch (outcome.toLowerCase()) {
      case 'successful':
        return 'bg-emerald-500';
      case 'delayed':
        return 'bg-amber-500';
      case 'missed':
        return 'bg-red-500';
      case 'critical incident':
        return 'bg-red-600';
      case 'concern raised':
        return 'bg-orange-500';
      default:
        return 'bg-slate-400';
    }
  };

  const getOutcomeIcon = (outcome?: string) => {
    if (!outcome) return <AlertTriangle className="w-3 h-3 text-white" />;
    
    switch (outcome.toLowerCase()) {
      case 'successful':
        return <CheckCircle className="w-3 h-3 text-white" />;
      case 'delayed':
        return <Clock className="w-3 h-3 text-white" />;
      case 'missed':
        return <XCircle className="w-3 h-3 text-white" />;
      case 'critical incident':
        return <AlertTriangle className="w-3 h-3 text-white" />;
      case 'concern raised':
        return <Clock className="w-3 h-3 text-white" />;
      default:
        return <AlertTriangle className="w-3 h-3 text-white" />;
    }
  };

  const getOutcomeBorderColor = (outcome?: string) => {
    if (!outcome) return 'border-slate-200';
    
    switch (outcome.toLowerCase()) {
      case 'successful':
        return 'border-emerald-200';
      case 'delayed':
        return 'border-amber-200';
      case 'missed':
        return 'border-red-200';
      case 'critical incident':
        return 'border-red-300';
      case 'concern raised':
        return 'border-orange-200';
      default:
        return 'border-slate-200';
    }
  };

  const getPositionPercent = (eventDate: string, eventIndex: number) => {
    if (!startDate || !endDate) return 50; // Center if no date range
    
    try {
      const start = new Date(startDate).getTime();
      const end = new Date(endDate).getTime();
      const event = new Date(eventDate).getTime();
      
      if (isNaN(start) || isNaN(end) || isNaN(event)) {
        return 50;
      }
      
      if (start === end) return 50; // Single date case
      
      // Calculate base position
      let basePosition = ((event - start) / (end - start)) * 100;
      
      // Ensure minimum spacing between events (at least 5% apart)
      const minSpacing = 5;
      const adjustedPosition = Math.max(8, Math.min(92, basePosition));
      
      return adjustedPosition;
    } catch (error) {
      return 50;
    }
  };

  // Function to distribute events with minimum spacing
  const getDistributedPositions = (events: TimelineEvent[]) => {
    if (events.length <= 1) return events.map(event => getPositionPercent(event.event_date, 0));
    
    const positions: number[] = [];
    const minSpacing = 5; // Minimum 5% between events
    
    for (let i = 0; i < events.length; i++) {
      const event = events[i];
      let basePosition = getPositionPercent(event.event_date, i);
      
      // Check if this position is too close to previous events
      let adjustedPosition = basePosition;
      for (let j = 0; j < i; j++) {
        const distance = Math.abs(adjustedPosition - positions[j]);
        if (distance < minSpacing) {
          // Move this event to maintain minimum spacing
          if (adjustedPosition > positions[j]) {
            adjustedPosition = positions[j] + minSpacing;
          } else {
            adjustedPosition = positions[j] - minSpacing;
          }
        }
      }
      
      // Ensure position stays within bounds
      adjustedPosition = Math.max(8, Math.min(92, adjustedPosition));
      positions.push(adjustedPosition);
    }
    
    return positions;
  };

  // Early return if no valid events
  if (!validEvents || validEvents.length === 0) {
    return (
      <div className="bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl p-6 border border-slate-200 shadow-sm">
        <div className="text-center text-slate-500 py-8">
          <Calendar className="w-12 h-12 mx-auto mb-4 text-slate-300" />
          <p className="text-lg font-medium">No Timeline Events Available</p>
          <p className="text-sm">This case review doesn't have any timeline events recorded.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-br from-slate-50 to-blue-50 rounded-xl p-6 border border-slate-200 shadow-sm">
      {/* Timeline Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-blue-100 rounded-lg">
            <Calendar className="w-5 h-5 text-blue-600" />
          </div>
          <div className="text-sm text-slate-700">
            <span className="font-semibold text-slate-900">Case Duration:</span>{' '}
            {startDate && endDate && (
              <>
                {new Date(startDate).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })} - {' '}
                {new Date(endDate).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}
              </>
            )}
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-xs text-slate-600 flex items-center space-x-4 bg-white/60 backdrop-blur-sm rounded-lg px-3 py-2 border border-slate-200">
            <span className="flex items-center">
              <span className="w-3 h-3 bg-emerald-500 rounded-full mr-2 shadow-sm"></span>Successful
            </span>
            <span className="flex items-center">
              <span className="w-3 h-3 bg-amber-500 rounded-full mr-2 shadow-sm"></span>Delayed
            </span>
            <span className="flex items-center">
              <span className="w-3 h-3 bg-red-500 rounded-full mr-2 shadow-sm"></span>Missed
            </span>
          </div>
        </div>
      </div>

      {/* Main Timeline */}
      <div className="relative h-32 bg-gradient-to-r from-white to-slate-50 rounded-xl border border-slate-200 overflow-visible mb-6 shadow-inner">
        {/* Timeline base line with gradient */}
        <div className="absolute top-1/2 left-0 right-0 h-1 bg-gradient-to-r from-blue-200 via-slate-300 to-blue-200 transform -translate-y-1/2"></div>
        
        {/* Timeline events */}
        {(() => {
          const distributedPositions = getDistributedPositions(sortedEvents);
          return sortedEvents.map((event, index) => {
            const position = distributedPositions[index];
            
            return (
              <div
                key={event.id}
                className="absolute top-1/2 transform -translate-y-1/2"
                style={{ left: `${position}%` }}
              >
                <div className="relative group cursor-pointer">
                  {/* Event marker with enhanced styling */}
                  <div className={`w-5 h-5 ${getOutcomeColor(event.impact)} rounded-full border-3 border-white shadow-lg hover:scale-125 transition-all duration-200 flex items-center justify-center ${getOutcomeBorderColor(event.impact)}`}>
                    {getOutcomeIcon(event.impact)}
                  </div>
                  
                  {/* Enhanced tooltip - positioned above the timeline to avoid cutoff */}
                  <div className="absolute bottom-12 left-1/2 transform -translate-x-1/2 bg-slate-900 text-white text-xs px-4 py-3 rounded-lg opacity-0 group-hover:opacity-100 transition-all duration-200 z-10 shadow-xl border border-slate-700 min-w-64 max-w-80">
                    <div className="font-medium mb-2 text-center">
                      {new Date(event.event_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                    </div>
                    <div className="text-slate-300 text-center leading-relaxed">{event.description}</div>
                    {/* Tooltip arrow pointing down to the event */}
                    <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-900"></div>
                  </div>
                </div>
              </div>
            );
          });
        })()}
      </div>

      {/* Enhanced Date markers */}
      {startDate && endDate && (
        <div className="flex justify-between text-sm text-slate-600 mt-6 px-2">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
            <span className="font-medium">{new Date(startDate).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="font-medium">{new Date(endDate).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}</span>
            <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
          </div>
        </div>
      )}

      {/* Timeline Legend & Controls */}
      <div className="mt-8 pt-6 border-t border-slate-200">
        <div className="flex items-center justify-between">
          <div className="text-sm text-slate-600 bg-blue-50 rounded-lg px-3 py-2 border border-blue-100">
            <strong className="text-blue-800">Key Insight:</strong> Timeline shows progression of events and agency responses.
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="text-slate-600 hover:text-blue-600 hover:bg-blue-50 transition-colors"
            onClick={() => setShowDetails(!showDetails)}
            data-testid="button-timeline-details"
          >
            <ChevronDown className={`w-4 h-4 mr-1 transition-transform duration-200 ${showDetails ? 'rotate-180' : ''}`} />
            {showDetails ? 'Hide' : 'Show'} Detailed Events
          </Button>
        </div>
      </div>

      {/* Enhanced Detailed Events */}
      {showDetails && (
        <div className="mt-6 space-y-4 animate-in slide-in-from-top-2 duration-300">
          {sortedEvents.map((event, index) => (
            <Card key={event.id} className={`border-l-4 ${getOutcomeBorderColor(event.impact)} hover:shadow-md transition-shadow duration-200`}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <Badge variant="outline" className="text-xs bg-slate-50">
                        <Calendar className="w-3 h-3 mr-1" />
                        {new Date(event.event_date).toLocaleDateString()}
                      </Badge>
                      {event.impact && (
                        <Badge className={`text-xs text-white ${getOutcomeColor(event.impact)} shadow-sm`}>
                          {getOutcomeIcon(event.impact)}
                          <span className="ml-1">{event.impact}</span>
                        </Badge>
                      )}
                    </div>
                    <h6 className="font-medium text-sm text-slate-900 mb-1">{event.description}</h6>
                    {event.impact && (
                      <p className="text-xs text-slate-600">Impact: {event.impact}</p>
                    )}
                  </div>
                  <Badge variant="secondary" className="text-xs ml-4 bg-slate-100 text-slate-700">
                    {event.event_type?.replace('_', ' ') || 'Event'}
                  </Badge>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

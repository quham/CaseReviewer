import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { Expand, Download, ChevronDown } from 'lucide-react';
import type { TimelineEvent } from '@/lib/search-api';

interface CaseTimelineProps {
  events: TimelineEvent[];
}

interface TimelineTrack {
  id: string;
  name: string;
  events: TimelineEvent[];
}

export function CaseTimeline({ events }: CaseTimelineProps) {
  const [showDetails, setShowDetails] = useState(false);

  // Group events by track
  const tracks: TimelineTrack[] = [
    {
      id: 'child_behavior',
      name: 'Child Behavior & Presentation',
      events: events.filter(e => e.eventType === 'child_behavior'),
    },
    {
      id: 'agency_action',
      name: 'Agency Actions & Decisions',
      events: events.filter(e => e.eventType === 'agency_action'),
    },
    {
      id: 'family_circumstance',
      name: 'Family Circumstances',
      events: events.filter(e => e.eventType === 'family_circumstance'),
    },
  ];

  // Get date range
  const sortedEvents = [...events].sort((a, b) => 
    new Date(a.eventDate).getTime() - new Date(b.eventDate).getTime()
  );
  const startDate = sortedEvents[0]?.eventDate;
  const endDate = sortedEvents[sortedEvents.length - 1]?.eventDate;

  const getOutcomeColor = (outcome?: string) => {
    switch (outcome) {
      case 'successful':
        return 'bg-success';
      case 'delayed':
        return 'bg-warning';
      case 'missed':
        return 'bg-danger';
      default:
        return 'bg-neutral-400';
    }
  };

  const getPositionPercent = (eventDate: string) => {
    if (!startDate || !endDate) return 0;
    
    const start = new Date(startDate).getTime();
    const end = new Date(endDate).getTime();
    const event = new Date(eventDate).getTime();
    
    return ((event - start) / (end - start)) * 100;
  };

  return (
    <div className="bg-neutral-50 rounded-lg p-6 border border-neutral-200">
      {/* Timeline Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="text-sm text-neutral-600">
          <span className="font-medium">Case Duration:</span>{' '}
          {startDate && endDate && (
            <>
              {new Date(startDate).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })} - {' '}
              {new Date(endDate).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}
            </>
          )}
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-xs text-neutral-600 flex items-center space-x-4">
            <span className="flex items-center">
              <span className="w-3 h-3 bg-success rounded-full mr-1"></span>Successful
            </span>
            <span className="flex items-center">
              <span className="w-3 h-3 bg-warning rounded-full mr-1"></span>Delayed
            </span>
            <span className="flex items-center">
              <span className="w-3 h-3 bg-danger rounded-full mr-1"></span>Missed
            </span>
          </div>
          <Button 
            variant="ghost" 
            size="sm"
            className="text-secondary hover:text-blue-600"
            data-testid="button-timeline-fullscreen"
          >
            <Expand className="w-4 h-4 mr-1" />
            Full Screen
          </Button>
          <Button 
            variant="ghost" 
            size="sm"
            className="text-secondary hover:text-blue-600"
            data-testid="button-timeline-export"
          >
            <Download className="w-4 h-4 mr-1" />
            Export Timeline
          </Button>
        </div>
      </div>

      {/* Multi-Track Timeline */}
      <div className="space-y-8">
        {tracks.map((track) => (
          <div key={track.id} className="relative">
            <div className="flex items-center mb-3">
              <h5 className="font-medium text-primary text-sm">{track.name}</h5>
            </div>
            <div className="relative h-16 bg-white rounded-lg border border-neutral-200 overflow-hidden">
              {/* Timeline base line */}
              <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-neutral-300 transform -translate-y-1/2"></div>
              
              {/* Timeline events */}
              {track.events.map((event, index) => {
                const position = getPositionPercent(event.eventDate);
                
                return (
                  <div
                    key={event.id}
                    className="absolute top-1/2 transform -translate-y-1/2"
                    style={{ left: `${Math.max(2, Math.min(95, position))}%` }}
                  >
                    <div className="relative group cursor-pointer">
                      <div className={`w-4 h-4 ${getOutcomeColor(event.outcome)} rounded-full border-2 border-white shadow-md hover:scale-110 transition-transform`} />
                      <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 bg-black text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10">
                        {new Date(event.eventDate).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}: {event.description}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Date markers */}
      {startDate && endDate && (
        <div className="flex justify-between text-xs text-neutral-500 mt-4 px-2">
          <span>{new Date(startDate).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}</span>
          <span>{new Date(endDate).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}</span>
        </div>
      )}

      {/* Timeline Legend & Controls */}
      <div className="mt-6 pt-4 border-t border-neutral-200">
        <div className="flex items-center justify-between">
          <div className="text-xs text-neutral-600">
            <strong>Key Insight:</strong> Timeline shows progression of events and agency responses across multiple tracks.
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="text-secondary hover:text-blue-600"
            onClick={() => setShowDetails(!showDetails)}
            data-testid="button-timeline-details"
          >
            <ChevronDown className={`w-4 h-4 mr-1 transition-transform ${showDetails ? 'rotate-180' : ''}`} />
            {showDetails ? 'Hide' : 'Show'} Detailed Events
          </Button>
        </div>
      </div>

      {/* Detailed Events */}
      {showDetails && (
        <div className="mt-4 space-y-3">
          {sortedEvents.map((event) => (
            <Card key={event.id} className="border-l-4 border-l-secondary">
              <CardContent className="p-4">
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center space-x-2 mb-1">
                      <Badge variant="outline" className="text-xs">
                        {new Date(event.eventDate).toLocaleDateString()}
                      </Badge>
                      {event.outcome && (
                        <Badge className={`text-xs text-white ${getOutcomeColor(event.outcome)}`}>
                          {event.outcome}
                        </Badge>
                      )}
                    </div>
                    <h6 className="font-medium text-sm">{event.description}</h6>
                    {event.details && (
                      <p className="text-xs text-neutral-600 mt-1">{event.details}</p>
                    )}
                  </div>
                  <Badge variant="secondary" className="text-xs ml-4">
                    {event.track.replace('_', ' ')}
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

import { 
  users, 
  caseReviews, 
  timelineEvents, 
  searches,
  type User, 
  type InsertUser,
  type CaseReview,
  type InsertCaseReview,
  type TimelineEvent,
  type InsertTimelineEvent,
  type Search,
  type InsertSearch,
  type SearchResult
} from "@shared/schema";
import { db } from "./db";
import { eq, and, desc, sql, or, ilike, inArray } from "drizzle-orm";

export interface IStorage {
  // User operations
  getUser(id: string): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;

  // Case review operations
  getCaseReview(id: string): Promise<CaseReview | undefined>;
  getCaseReviews(): Promise<CaseReview[]>;
  createCaseReview(caseReview: InsertCaseReview): Promise<CaseReview>;
  searchCaseReviews(query: string, filters?: {
    childAge?: string;
    riskType?: string;
    outcome?: string;
    reviewDate?: string;
  }): Promise<SearchResult[]>;

  // Timeline operations
  getTimelineEvents(caseReviewId: string): Promise<TimelineEvent[]>;
  createTimelineEvent(event: InsertTimelineEvent): Promise<TimelineEvent>;

  // Search history
  createSearch(search: InsertSearch): Promise<Search>;
  getUserSearches(userId: string): Promise<Search[]>;
}

export class DatabaseStorage implements IStorage {
  async getUser(id: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user || undefined;
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.username, username));
    return user || undefined;
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const [user] = await db
      .insert(users)
      .values(insertUser)
      .returning();
    return user;
  }

  async getCaseReview(id: string): Promise<CaseReview | undefined> {
    const [caseReview] = await db.select().from(caseReviews).where(eq(caseReviews.id, id));
    return caseReview || undefined;
  }

  async getCaseReviews(): Promise<CaseReview[]> {
    return await db.select().from(caseReviews).orderBy(desc(caseReviews.reviewDate));
  }

  async createCaseReview(insertCaseReview: InsertCaseReview): Promise<CaseReview> {
    const [caseReview] = await db
      .insert(caseReviews)
      .values([insertCaseReview])
      .returning();
    return caseReview;
  }

  async searchCaseReviews(query: string, filters?: {
    childAge?: string;
    riskType?: string;
    outcome?: string;
    reviewDate?: string;
  }): Promise<SearchResult[]> {
    let whereConditions = [];

    // Text search across multiple fields
    if (query.trim()) {
      whereConditions.push(
        or(
          ilike(caseReviews.title, `%${query}%`),
          ilike(caseReviews.summary, `%${query}%`),
          ilike(caseReviews.content, `%${query}%`)
        )
      );
    }

    // Apply filters
    if (filters?.childAge && filters.childAge !== 'Any age') {
      const ageRanges = {
        '0-5 years': [0, 5],
        '6-11 years': [6, 11],
        '12-17 years': [12, 17]
      };
      const range = ageRanges[filters.childAge as keyof typeof ageRanges];
      if (range) {
        whereConditions.push(
          and(
            sql`${caseReviews.childAge} >= ${range[0]}`,
            sql`${caseReviews.childAge} <= ${range[1]}`
          )
        );
      }
    }

    if (filters?.outcome && filters.outcome !== 'All outcomes') {
      whereConditions.push(eq(caseReviews.outcome, filters.outcome));
    }

    const whereClause = whereConditions.length > 0 ? and(...whereConditions) : undefined;

    const results = await db
      .select()
      .from(caseReviews)
      .where(whereClause)
      .orderBy(desc(caseReviews.reviewDate));

    // Get timeline events for each case review
    const searchResults: SearchResult[] = [];
    for (const caseReview of results) {
      const events = await this.getTimelineEvents(caseReview.id);
      
      // Calculate relevance score (simplified)
      let relevanceScore = 50;
      if (query.trim()) {
        const queryLower = query.toLowerCase();
        if (caseReview.title.toLowerCase().includes(queryLower)) relevanceScore += 30;
        if (caseReview.summary.toLowerCase().includes(queryLower)) relevanceScore += 20;
      }
      relevanceScore = Math.min(100, relevanceScore);

      // Extract key matches
      const keyMatches: string[] = [];
      if (filters?.childAge && filters.childAge !== 'Any age') {
        keyMatches.push(`Child age matches ${filters.childAge}`);
      }
      if (filters?.riskType && filters.riskType !== 'All types') {
        keyMatches.push(`Risk type: ${filters.riskType}`);
      }

      searchResults.push({
        ...caseReview,
        relevanceScore,
        keyMatches,
        timelineEvents: events,
      });
    }

    return searchResults.sort((a, b) => b.relevanceScore - a.relevanceScore);
  }

  async getTimelineEvents(caseReviewId: string): Promise<TimelineEvent[]> {
    return await db
      .select()
      .from(timelineEvents)
      .where(eq(timelineEvents.caseReviewId, caseReviewId))
      .orderBy(timelineEvents.eventDate);
  }

  async createTimelineEvent(insertEvent: InsertTimelineEvent): Promise<TimelineEvent> {
    const [event] = await db
      .insert(timelineEvents)
      .values(insertEvent)
      .returning();
    return event;
  }

  async createSearch(insertSearch: InsertSearch): Promise<Search> {
    const [search] = await db
      .insert(searches)
      .values([insertSearch])
      .returning();
    return search;
  }

  async getUserSearches(userId: string): Promise<Search[]> {
    return await db
      .select()
      .from(searches)
      .where(eq(searches.userId, userId))
      .orderBy(desc(searches.searchedAt));
  }
}

export const storage = new DatabaseStorage();

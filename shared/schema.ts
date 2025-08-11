import { sql } from "drizzle-orm";
import { pgTable, text, varchar, timestamp, integer, jsonb, boolean } from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
  role: text("role").notNull().default("social_worker"),
  name: text("name").notNull(),
  organization: text("organization"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const caseReviews = pgTable("case_reviews", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  title: text("title").notNull(),
  summary: text("summary").notNull(),
  content: text("content").notNull(),
  childAge: integer("child_age"),
  riskTypes: jsonb("risk_types").$type<string[]>().default([]),
  outcome: text("outcome"),
  reviewDate: timestamp("review_date"),
  agencies: jsonb("agencies").$type<string[]>().default([]),
  warningSignsEarly: jsonb("warning_signs_early").$type<string[]>().default([]),
  riskFactors: jsonb("risk_factors").$type<string[]>().default([]),
  barriers: jsonb("barriers").$type<string[]>().default([]),
  relationshipModel: jsonb("relationship_model").$type<{
    familyStructure: string;
    professionalNetwork: string;
    supportSystems: string;
    powerDynamics: string;
  }>(),
  documentUrl: text("document_url"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const timelineEvents = pgTable("timeline_events", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  caseReviewId: varchar("case_review_id").notNull().references(() => caseReviews.id),
  eventDate: timestamp("event_date").notNull(),
  eventType: text("event_type").notNull(), // 'child_behavior', 'agency_action', 'family_circumstance'
  description: text("description").notNull(),
  outcome: text("outcome"), // 'successful', 'delayed', 'missed'
  details: text("details"),
  track: text("track").notNull(), // timeline track identifier
});

export const searches = pgTable("searches", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  query: text("query").notNull(),
  filters: jsonb("filters").$type<{
    childAge?: string;
    riskType?: string;
    outcome?: string;
    reviewDate?: string;
  }>(),
  resultsCount: integer("results_count"),
  searchedAt: timestamp("searched_at").defaultNow(),
});

export const usersRelations = relations(users, ({ many }) => ({
  searches: many(searches),
}));

export const caseReviewsRelations = relations(caseReviews, ({ many }) => ({
  timelineEvents: many(timelineEvents),
}));

export const timelineEventsRelations = relations(timelineEvents, ({ one }) => ({
  caseReview: one(caseReviews, {
    fields: [timelineEvents.caseReviewId],
    references: [caseReviews.id],
  }),
}));

export const searchesRelations = relations(searches, ({ one }) => ({
  user: one(users, {
    fields: [searches.userId],
    references: [users.id],
  }),
}));

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
  name: true,
  organization: true,
});

export const insertCaseReviewSchema = createInsertSchema(caseReviews).omit({
  id: true,
  createdAt: true,
});

export const insertTimelineEventSchema = createInsertSchema(timelineEvents).omit({
  id: true,
});

export const insertSearchSchema = createInsertSchema(searches).omit({
  id: true,
  searchedAt: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type InsertCaseReview = z.infer<typeof insertCaseReviewSchema>;
export type CaseReview = typeof caseReviews.$inferSelect;
export type InsertTimelineEvent = z.infer<typeof insertTimelineEventSchema>;
export type TimelineEvent = typeof timelineEvents.$inferSelect;
export type InsertSearch = z.infer<typeof insertSearchSchema>;
export type Search = typeof searches.$inferSelect;

export interface SearchResult extends CaseReview {
  relevanceScore: number;
  keyMatches: string[];
  aiAdvice?: string;
  timelineEvents: TimelineEvent[];
}

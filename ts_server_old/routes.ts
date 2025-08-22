import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertUserSchema, insertSearchSchema } from "@shared/schema";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import { generateCaseAdvice } from "./services/openai";

const JWT_SECRET = process.env.JWT_SECRET || "nspcc-case-review-secret";

interface AuthenticatedRequest extends Request {
  user?: { id: string; username: string; role: string };
}

// Middleware to verify JWT token
const authenticateToken = (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.sendStatus(401);
  }

  jwt.verify(token, JWT_SECRET, (err: any, user: any) => {
    if (err) return res.sendStatus(403);
    req.user = user;
    next();
  });
};

export async function registerRoutes(app: Express): Promise<Server> {
  // Authentication routes
  app.post("/api/register", async (req, res) => {
    try {
      const userData = insertUserSchema.parse(req.body);
      const hashedPassword = await bcrypt.hash(userData.password, 10);
      
      const user = await storage.createUser({
        ...userData,
        password: hashedPassword,
      });

      const token = jwt.sign(
        { id: user.id, username: user.username, role: user.role },
        JWT_SECRET
      );

      res.json({ token, user: { id: user.id, username: user.username, name: user.name, role: user.role } });
    } catch (error) {
      console.error("Registration error:", error);
      res.status(400).json({ error: "Registration failed" });
    }
  });

  app.post("/api/login", async (req, res) => {
    try {
      const { username, password } = req.body;
      
      const user = await storage.getUserByUsername(username);
      if (!user || !(await bcrypt.compare(password, user.password))) {
        return res.status(401).json({ error: "Invalid credentials" });
      }

      const token = jwt.sign(
        { id: user.id, username: user.username, role: user.role },
        JWT_SECRET
      );

      res.json({ token, user: { id: user.id, username: user.username, name: user.name, role: user.role } });
    } catch (error) {
      console.error("Login error:", error);
      res.status(400).json({ error: "Login failed" });
    }
  });

  // Protected routes
  app.use("/api/protected", authenticateToken);

  app.get("/api/protected/me", (req: AuthenticatedRequest, res) => {
    res.json({ user: req.user });
  });

  app.post("/api/protected/search", async (req: AuthenticatedRequest, res) => {
    try {
      const { query, filters } = req.body;
      
      if (!query || query.trim().length === 0) {
        return res.status(400).json({ error: "Search query is required" });
      }

      // Perform search
      const results = await storage.searchCaseReviews(query, filters);

      // Generate AI advice for top results
      const resultsWithAdvice = await Promise.all(
        results.slice(0, 3).map(async (result) => {
          try {
            const aiAdvice = await generateCaseAdvice(query, result);
            return { ...result, aiAdvice };
          } catch (error) {
            console.error("AI advice generation error:", error);
            return result;
          }
        })
      );

      // Add remaining results without AI advice
      const finalResults = [...resultsWithAdvice, ...results.slice(3)];

      // Save search to history
      if (req.user) {
        await storage.createSearch({
          userId: req.user.id,
          query,
          filters,
          resultsCount: results.length,
        });
      }

      res.json({
        results: finalResults,
        totalCount: results.length,
        searchTime: Math.random() * 1000 + 500, // Simulated search time
      });
    } catch (error) {
      console.error("Search error:", error);
      res.status(500).json({ error: "Search failed" });
    }
  });

  app.get("/api/protected/case-reviews/:id", async (req: AuthenticatedRequest, res) => {
    try {
      const caseReview = await storage.getCaseReview(req.params.id);
      if (!caseReview) {
        return res.status(404).json({ error: "Case review not found" });
      }

      const timelineEvents = await storage.getTimelineEvents(caseReview.id);
      res.json({ ...caseReview, timelineEvents });
    } catch (error) {
      console.error("Case review fetch error:", error);
      res.status(500).json({ error: "Failed to fetch case review" });
    }
  });

  app.get("/api/protected/search-history", async (req: AuthenticatedRequest, res) => {
    try {
      if (!req.user) {
        return res.status(401).json({ error: "User not authenticated" });
      }

      const searches = await storage.getUserSearches(req.user.id);
      res.json(searches);
    } catch (error) {
      console.error("Search history error:", error);
      res.status(500).json({ error: "Failed to fetch search history" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}

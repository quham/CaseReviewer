# NSPCC Case Review Search System

## Overview

The NSPCC Case Review Search System is a specialized web application designed for social workers to search and analyze historical case reviews for evidence-based insights. The system enables professionals to input natural language descriptions of current cases and receive relevant case review matches with AI-generated advice, interactive timelines, and comprehensive analysis tools. Built as a secure, professional platform, it combines sophisticated search capabilities with actionable recommendations to support early intervention and risk mitigation strategies.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React 18 with TypeScript, utilizing modern hooks and functional components
- **Build Tool**: Vite for fast development and optimized production builds
- **UI Framework**: shadcn/ui components built on Radix UI primitives for accessibility and consistency
- **Styling**: Tailwind CSS with custom design tokens and CSS variables for theming
- **State Management**: Zustand for authentication state with persistence middleware
- **Data Fetching**: TanStack Query (React Query) for server state management and caching
- **Routing**: Wouter for lightweight client-side routing
- **Forms**: React Hook Form with Zod validation for type-safe form handling

### Backend Architecture
- **Runtime**: Node.js with Express.js REST API server
- **Language**: TypeScript with ES modules throughout the application
- **Authentication**: JWT-based stateless authentication with bcrypt password hashing
- **Session Management**: Secure token-based approach for maintaining user sessions
- **API Design**: RESTful endpoints with consistent error handling and logging middleware
- **Development Setup**: Hot module replacement with Vite middleware integration

### Data Storage Solutions
- **Primary Database**: PostgreSQL with Neon serverless driver for scalability
- **ORM**: Drizzle ORM for type-safe database operations and schema management
- **Schema Design**: Comprehensive relational model including users, case reviews, timeline events, and search history
- **Migrations**: Drizzle Kit for database schema versioning and deployment
- **Connection Pooling**: Neon serverless pool for efficient database connections

### Authentication and Authorization
- **Strategy**: JWT tokens with configurable secret key for secure authentication
- **Password Security**: bcrypt hashing with appropriate salt rounds
- **Role-Based Access**: User roles (primarily social_worker) for future authorization expansion
- **Token Management**: Client-side token storage with automatic authentication checking
- **Protected Routes**: Middleware-based route protection for API endpoints

### Search and AI Integration
- **Search Engine**: Sophisticated text-based search with relevance scoring and filtering
- **AI Integration**: OpenAI GPT-4o integration for generating contextual case advice
- **Natural Language Processing**: Support for complex case description input with multi-criteria matching
- **Timeline Visualization**: Interactive timeline components for case event tracking
- **Result Presentation**: Structured search results with relevance scoring and key match highlighting

## External Dependencies

### Database Services
- **Neon Database**: Serverless PostgreSQL platform providing scalable database hosting
- **Connection Management**: WebSocket-based connections for real-time capabilities

### AI and Machine Learning
- **OpenAI API**: GPT-4o model integration for generating professional case advice and recommendations
- **Natural Language Processing**: Contextual analysis of case descriptions and historical reviews

### UI Component Libraries
- **Radix UI**: Comprehensive collection of accessible, unstyled React components including dialogs, dropdowns, navigation, and form controls
- **Lucide React**: Icon library providing consistent iconography throughout the application
- **React Day Picker**: Calendar component for date selection and timeline interactions

### Development and Build Tools
- **Vite**: Modern build tool with fast hot module replacement and optimized production builds
- **Replit Integration**: Development environment plugins for seamless cloud-based development
- **PostCSS**: CSS processing with Tailwind CSS and Autoprefixer plugins

### Authentication and Security
- **bcrypt**: Industry-standard password hashing library for secure credential storage
- **jsonwebtoken**: JWT implementation for stateless authentication tokens
- **Security Headers**: CORS and security middleware for API protection

### State Management and Data Fetching
- **TanStack Query**: Powerful data synchronization for React with caching, background updates, and optimistic updates
- **Zustand**: Lightweight state management with TypeScript support and persistence capabilities

### Form Handling and Validation
- **React Hook Form**: Performant forms library with minimal re-renders
- **Zod**: TypeScript-first schema validation for runtime type checking
- **Hookform Resolvers**: Integration layer between React Hook Form and validation libraries

### Utility Libraries
- **date-fns**: Modern date manipulation library for timeline and date formatting
- **class-variance-authority**: Utility for creating variant-based component APIs
- **clsx**: Utility for constructing className strings conditionally
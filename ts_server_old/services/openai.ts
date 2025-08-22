import OpenAI from "openai";
import type { SearchResult } from "@shared/schema";

// the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
const openai = new OpenAI({ 
  apiKey: process.env.OPENAI_API_KEY || process.env.OPENAI_API_KEY_ENV_VAR || "default_key"
});

export async function generateCaseAdvice(
  userQuery: string, 
  caseReview: SearchResult
): Promise<string> {
  try {
    const prompt = `You are an expert social work advisor specializing in child protection cases. 

Based on the user's current case query and this relevant historical case review, provide specific, actionable advice.

USER'S CURRENT CASE:
${userQuery}

RELEVANT HISTORICAL CASE REVIEW:
Title: ${caseReview.title}
Summary: ${caseReview.summary}
Child Age: ${caseReview.childAge || 'Not specified'}
Risk Types: ${caseReview.riskTypes ? caseReview.riskTypes.join(', ') : 'Not specified'}
Outcome: ${caseReview.outcome || 'Not specified'}
Warning Signs: ${caseReview.warningSignsEarly ? caseReview.warningSignsEarly.join(', ') : 'Not specified'}
Risk Factors: ${caseReview.riskFactors ? caseReview.riskFactors.join(', ') : 'Not specified'}
Barriers: ${caseReview.barriers ? caseReview.barriers.join(', ') : 'Not specified'}

Please provide advice in the following format:

IMMEDIATE ACTIONS:
- [Specific action 1]
- [Specific action 2]
- [Specific action 3]

ASSESSMENT QUESTIONS:
- [Key question 1]
- [Key question 2] 
- [Key question 3]

EARLY WARNING SIGNS TO MONITOR:
- [Warning sign 1]
- [Warning sign 2]
- [Warning sign 3]

MULTI-AGENCY CONSIDERATIONS:
- [Agency coordination point 1]
- [Agency coordination point 2]

Keep advice practical, specific, and directly applicable to the user's case situation. Focus on evidence-based interventions that were successful or could have prevented issues in the historical case.`;

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: "You are an expert social work advisor. Provide clear, actionable advice based on case review analysis. Be professional, empathetic, and evidence-based."
        },
        {
          role: "user",
          content: prompt
        }
      ],
      max_tokens: 800,
      temperature: 0.3
    });

    return response.choices[0].message.content || "Unable to generate advice at this time.";
  } catch (error) {
    console.error("OpenAI API error:", error);
    throw new Error("Failed to generate AI advice. Please try again later.");
  }
}

export async function generateKeyMatches(
  userQuery: string,
  caseReview: SearchResult
): Promise<string[]> {
  try {
    const prompt = `Analyze why this case review matches the user's query and identify the key matching elements.

USER QUERY: ${userQuery}

CASE REVIEW:
- Title: ${caseReview.title}
- Child Age: ${caseReview.childAge}
- Risk Types: ${caseReview.riskTypes ? caseReview.riskTypes.join(', ') : 'Not specified'}
- Summary: ${caseReview.summary}

Return a JSON object with an array of key matches. Format:
{
  "matches": [
    "Similar age child (X years) with [specific behavior]",
    "Matching risk factors: [specific factors]",
    "[Other specific similarities]"
  ]
}`;

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system", 
          content: "You are analyzing case similarities for social workers. Identify specific matching elements between cases."
        },
        {
          role: "user",
          content: prompt
        }
      ],
      response_format: { type: "json_object" },
      max_tokens: 300,
      temperature: 0.2
    });

    const result = JSON.parse(response.choices[0].message.content || '{"matches": []}');
    return result.matches || [];
  } catch (error) {
    console.error("Key matches generation error:", error);
    return [`Case involves similar risk factors and circumstances`];
  }
}

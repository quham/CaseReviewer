import { storage } from "./storage";
import bcrypt from "bcrypt";

export async function seedDatabase() {
  try {
    console.log("Seeding database with initial data...");

    // Create demo user
    const hashedPassword = await bcrypt.hash("socialworker2024", 10);
    const user = await storage.createUser({
      username: "sarah.johnson",
      password: hashedPassword,
      name: "Sarah Johnson",
      organization: "Riverside Children's Services"
    });

    // Create sample case reviews
    const caseReview1 = await storage.createCaseReview({
      title: "Domestic Violence & New Partner Concerns - Child Behavioral Regression",
      summary: "Case involving 6-year-old child showing regression behaviors in context of mother's new partner with anger management issues and historical domestic violence. Successful intervention through multi-agency approach and housing support.",
      content: `This case review examines the intervention in a family where Emma (6) lived with her mother Sarah (24) and mother's new partner David (31). The case highlights the importance of recognizing behavioral regression as an early warning sign and the critical role of housing stability in safety planning.

Background:
Emma had been living with her mother following separation from her father 18 months prior due to domestic violence. Father maintained supervised monthly contact. Mother entered new relationship with David who moved in after 8 months of dating.

Initial Concerns:
- Emma began bedwetting after being dry for 2 years
- School reported withdrawal and reluctance to go home
- Mother reported partner "loses temper" with Emma, particularly around bedtime
- Mother expressed feeling "caught in the middle" and worried about housing if partner left

Timeline of Events:
January 2023: David moved in, Emma's bedwetting started
February 2023: School raised concerns about Emma's behavior
March 2023: Initial assessment begun (2-week delay from referral)
April 2023: Multi-agency meeting, housing concerns identified
June 2023: Safety plan implemented, housing support arranged
September 2023: Case successfully closed with ongoing monitoring

Key Learning Points:
1. Behavioral regression in previously toilet-trained children requires immediate assessment
2. Mother's housing fears created barriers to disclosure
3. Early multi-agency coordination was crucial to success
4. Housing support was fundamental to safety planning effectiveness

The case demonstrates successful intervention where early warning signs were recognized and addressed through coordinated multi-agency response.`,
      childAge: 6,
      riskTypes: ["Domestic violence", "Emotional abuse", "New partner concerns"],
      outcome: "Successful intervention",
      reviewDate: new Date("2023-09-15"),
      agencies: ["Children's Services", "Police", "School", "Health Visitor", "Housing"],
      warningSignsEarly: [
        "Regression in toileting behavior after 2 years being dry",
        "School reports of withdrawal and reluctance to go home", 
        "Behavioral changes coinciding with new partner moving in",
        "Mother's conflicted feelings about relationship"
      ],
      riskFactors: [
        "New partner with reported anger management issues",
        "Historical domestic violence between parents",
        "Housing insecurity and financial dependence", 
        "Limited support network for mother",
        "Child caught between competing loyalties"
      ],
      barriers: [
        "Mother's fear of homelessness if relationship ended",
        "Partner's resistance to social services involvement",
        "Limited availability of therapeutic support for Emma",
        "Complex contact arrangements with biological father"
      ],
      relationshipModel: {
        familyStructure: "Single mother, new partner, child, non-resident father with contact",
        professionalNetwork: "Social worker (lead), police liaison, school SENCO, health visitor, housing officer",
        supportSystems: "Limited family support, strong school relationship, emerging partner support",
        powerDynamics: "Mother dependent on partner for housing; father maintains legal contact rights; child lacks voice in arrangements"
      },
      documentUrl: "https://example.com/case-reviews/NSPCC-CR-2023-047"
    });

    // Create timeline events for case review 1
    await storage.createTimelineEvent({
      caseReviewId: caseReview1.id,
      eventDate: new Date("2023-01-03"),
      eventType: "family_circumstance", 
      description: "New partner moved into family home",
      outcome: "missed",
      details: "David moved in after 8 months of dating. No prior risk assessment conducted.",
      track: "family_circumstances"
    });

    await storage.createTimelineEvent({
      caseReviewId: caseReview1.id,
      eventDate: new Date("2023-01-15"),
      eventType: "child_behavior",
      description: "Emma started bedwetting again after 2 years being dry",
      outcome: "delayed",
      details: "Significant regression behavior noted but not immediately reported to services.",
      track: "child_behavior"
    });

    await storage.createTimelineEvent({
      caseReviewId: caseReview1.id,
      eventDate: new Date("2023-02-08"),
      eventType: "child_behavior", 
      description: "School reports Emma's withdrawal and reluctance to go home",
      outcome: "delayed",
      details: "Teachers noted behavioral changes and contacted social services.",
      track: "child_behavior"
    });

    await storage.createTimelineEvent({
      caseReviewId: caseReview1.id,
      eventDate: new Date("2023-03-05"),
      eventType: "agency_action",
      description: "Initial assessment started (2 weeks after referral)",
      outcome: "delayed", 
      details: "Assessment delayed due to capacity issues and initial categorization as 'lower priority'.",
      track: "agency_actions"
    });

    await storage.createTimelineEvent({
      caseReviewId: caseReview1.id,
      eventDate: new Date("2023-04-01"),
      eventType: "family_circumstance",
      description: "Mother raised housing concerns and fears about relationship ending",
      outcome: "successful",
      details: "Assessment identified housing as key barrier to safety planning.",
      track: "family_circumstances"
    });

    await storage.createTimelineEvent({
      caseReviewId: caseReview1.id,
      eventDate: new Date("2023-04-12"), 
      eventType: "agency_action",
      description: "Multi-agency meeting arranged and conducted",
      outcome: "successful",
      details: "Effective coordination between children's services, police, school, and housing.",
      track: "agency_actions"
    });

    await storage.createTimelineEvent({
      caseReviewId: caseReview1.id,
      eventDate: new Date("2023-06-15"),
      eventType: "agency_action",
      description: "Comprehensive safety plan implemented",
      outcome: "successful",
      details: "Plan included housing support, therapeutic services for Emma, and partner engagement.",
      track: "agency_actions"
    });

    await storage.createTimelineEvent({
      caseReviewId: caseReview1.id,
      eventDate: new Date("2023-07-22"),
      eventType: "child_behavior",
      description: "Significant behavioral improvements noted by school and mother",
      outcome: "successful", 
      details: "Emma's bedwetting reduced, increased engagement at school, appears more settled.",
      track: "child_behavior"
    });

    await storage.createTimelineEvent({
      caseReviewId: caseReview1.id,
      eventDate: new Date("2023-08-10"),
      eventType: "family_circumstance",
      description: "Housing support secured and family stability improved",
      outcome: "successful",
      details: "Alternative housing options identified, reducing mother's dependency fears.",
      track: "family_circumstances"
    });

    // Create second case review
    const caseReview2 = await storage.createCaseReview({
      title: "Multi-Agency Response to Family Violence - Delayed Recognition",
      summary: "Case involving delayed agency response to escalating domestic violence affecting two children. Highlights importance of early intervention and multi-agency coordination in complex family situations.",
      content: `Case review examining response to domestic violence in family with Jake (14) and Mia (11). Father's violence escalated over 6 months before effective intervention. Case demonstrates both system failures and eventual successful multi-agency response.

The case involved multiple warning signs that were not initially recognized as part of a pattern, leading to delayed intervention and increased risk to the children.`,
      childAge: 12, // Average of the two children
      riskTypes: ["Domestic violence", "Neglect", "Emotional abuse"],
      outcome: "Delayed intervention - eventual success",
      reviewDate: new Date("2023-11-20"),
      agencies: ["Children's Services", "Police", "School", "Mental Health Services"],
      warningSignsEarly: [
        "Increased school absences for both children",
        "Aggressive behavior from older child at school", 
        "Younger child taking on caring responsibilities",
        "Neighbor complaints about noise and disturbances"
      ],
      riskFactors: [
        "Escalating domestic violence",
        "Mother's alcohol use following separation",
        "Children's emotional and behavioral difficulties",
        "Social isolation and limited support network"
      ],
      barriers: [
        "Initial categorization as 'lower risk' case",
        "Lack of communication between agencies",
        "Mother's reluctance to engage due to fear and shame",
        "Limited therapeutic resources for adolescents"
      ],
      relationshipModel: {
        familyStructure: "Recently separated parents, two children, limited extended family support",
        professionalNetwork: "Social worker, police domestic violence specialist, school counselors",
        supportSystems: "Grandmother providing some support, school as protective factor",
        powerDynamics: "Father using children to maintain control, mother struggling with independence"
      },
      documentUrl: "https://example.com/case-reviews/NSPCC-CR-2023-089"
    });

    // Create third case review  
    const caseReview3 = await storage.createCaseReview({
      title: "Child Behavioral Regression in Domestic Violence Context - Early Intervention",
      summary: "Successful early intervention case where behavioral regression indicators were quickly recognized and addressed, preventing escalation of harm to young child in domestic violence situation.",
      content: `This case demonstrates effective early intervention when a 5-year-old child showed regression behaviors following parental separation and new domestic violence concerns. Quick recognition and response prevented escalation.`,
      childAge: 5,
      riskTypes: ["Domestic violence", "Behavioral regression"],
      outcome: "Successful early intervention", 
      reviewDate: new Date("2023-08-10"),
      agencies: ["Children's Services", "Health Visitor", "Nursery School"],
      warningSignsEarly: [
        "Sleep disturbances and nightmares",
        "Regression in language development",
        "Increased clinginess and separation anxiety",
        "Physical complaints with no medical cause"
      ],
      riskFactors: [
        "Recent parental separation",
        "New domestic violence concerns",
        "Child's young age and vulnerability",
        "Mother's mental health difficulties"
      ],
      barriers: [
        "Mother's initial resistance to support",
        "Limited nursery hours affecting assessment opportunities",
        "Waiting lists for specialist therapeutic support"
      ],
      relationshipModel: {
        familyStructure: "Single mother, young child, absent father",
        professionalNetwork: "Social worker, health visitor, nursery key worker",
        supportSystems: "Strong maternal grandmother support, engaged nursery",
        powerDynamics: "Mother empowered through early support, child's voice heard through play"
      },
      documentUrl: "https://example.com/case-reviews/NSPCC-CR-2023-062"
    });

    console.log("Database seeded successfully!");
    console.log(`Created user: ${user.username}`);
    console.log(`Created ${3} case reviews with timeline events`);

  } catch (error) {
    console.error("Error seeding database:", error);
    throw error;
  }
}

// Run seeding if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  seedDatabase()
    .then(() => {
      console.log("Seeding completed successfully");
      process.exit(0);
    })
    .catch((error) => {
      console.error("Seeding failed:", error);
      process.exit(1);
    });
}

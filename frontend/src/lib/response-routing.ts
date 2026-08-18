/**
 * Emergency-response routing already implemented in the SentinelAI backend.
 * Email is the only implemented notification channel.
 */
export interface ResponseRoute {
  department: string;
  priority: "CRITICAL" | "HIGH" | "NONE";
  notifies: boolean;
}

export function responseRoute(classification: string | null | undefined): ResponseRoute {
  switch (classification) {
    case "Fire":
      return { department: "Fire & Rescue", priority: "CRITICAL", notifies: true };
    case "Fight":
      return { department: "Police / Security", priority: "HIGH", notifies: true };
    case "Road Accident":
      return { department: "Emergency Medical / Traffic", priority: "HIGH", notifies: true };
    case "Normal":
      return { department: "No emergency response", priority: "NONE", notifies: false };
    default:
      return { department: "Unclassified", priority: "NONE", notifies: false };
  }
}

"""
SentinelAI - Emergency Notification Service

Purpose
-------
Routes detected SentinelAI incidents to the appropriate emergency
department and sends an email notification.

Current notification provider:
    Resend

Current notification channel:
    Email

Incident routing:
    Fire          -> Fire & Rescue
    Fight         -> Police / Security
    Road Accident -> Emergency Medical / Traffic
    Normal        -> No notification

This module is intentionally independent from the Qwen inference
engine. The inference engine only detects/classifies the incident.
FastAPI calls this module after creating an incident.

Environment variables required:

    RESEND_API_KEY=your_resend_api_key
    ALERT_EMAIL_TO=your_verified_recipient_email

For local/demo use, Resend's onboarding sender is used:

    onboarding@resend.dev

Later, this can be replaced with a verified custom domain sender.
"""

from __future__ import annotations

import os
from html import escape
from typing import Any

import resend
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

# Load variables from:
#
# C:\SentinelAI_Qwen\.env
#
load_dotenv()


RESEND_API_KEY = os.getenv(
    "RESEND_API_KEY"
)

ALERT_EMAIL_TO = os.getenv(
    "ALERT_EMAIL_TO"
)


# ============================================================
# INCIDENT ROUTING
# ============================================================

DEPARTMENT_ROUTING: dict[
    str,
    dict[str, str],
] = {
    "Fire": {
        "department": "Fire & Rescue",
        "priority": "CRITICAL",
    },
    "Fight": {
        "department": "Police / Security",
        "priority": "HIGH",
    },
    "Road Accident": {
        "department": "Emergency Medical / Traffic",
        "priority": "HIGH",
    },
}


# ============================================================
# CONFIGURATION
# ============================================================

def email_configured() -> bool:
    """
    Check whether the required Resend configuration exists.
    """

    return bool(
        RESEND_API_KEY
        and ALERT_EMAIL_TO
    )


# ============================================================
# INCIDENT ROUTING
# ============================================================

def route_incident(
    incident_type: str,
) -> dict[str, str]:
    """
    Determine the responsible department and priority.

    Parameters
    ----------
    incident_type:
        SentinelAI classification.

    Returns
    -------
    dict
        Department and response priority.
    """

    routing = DEPARTMENT_ROUTING.get(
        incident_type
    )

    if routing is None:
        return {
            "department": "None",
            "priority": "LOW",
        }

    return routing.copy()


# ============================================================
# EMAIL CONTENT
# ============================================================

def build_incident_email(
    incident: dict[str, Any],
) -> dict[str, str]:
    """
    Build the subject and HTML body for an incident email.
    """

    incident_type = str(
        incident.get(
            "incident_type",
            "Unknown",
        )
    )

    threat_level = str(
        incident.get(
            "threat_level",
            "UNKNOWN",
        )
    )

    incident_id = str(
        incident.get(
            "id",
            "UNKNOWN",
        )
    )

    department = str(
        incident.get(
            "department",
            "Unknown",
        )
    )

    summary = str(
        incident.get(
            "summary",
            "",
        )
    )

    recommended_action = str(
        incident.get(
            "recommended_action",
            "",
        )
    )

    location = str(
        incident.get(
            "location",
            "Unknown",
        )
    )

    date_time = str(
        incident.get(
            "date_time",
            "Unknown",
        )
    )

    # --------------------------------------------------------
    # Escape user/model-generated values before putting them
    # into HTML.
    # --------------------------------------------------------

    incident_type_html = escape(
        incident_type
    )

    threat_level_html = escape(
        threat_level
    )

    incident_id_html = escape(
        incident_id
    )

    department_html = escape(
        department
    )

    summary_html = escape(
        summary
    )

    recommended_action_html = escape(
        recommended_action
    )

    location_html = escape(
        location
    )

    date_time_html = escape(
        date_time
    )

    # --------------------------------------------------------
    # EMAIL SUBJECT
    # --------------------------------------------------------

    subject = (
        f"[SentinelAI] "
        f"{threat_level} INCIDENT - "
        f"{incident_type} - "
        f"{incident_id}"
    )

    # --------------------------------------------------------
    # EMAIL HTML
    # --------------------------------------------------------

    html = f"""
<!DOCTYPE html>

<html>

<head>

    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width,
                 initial-scale=1.0"
    >

    <title>
        SentinelAI Incident Alert
    </title>

</head>


<body
    style="
        margin:0;
        padding:0;
        background:#f4f4f5;
        font-family:
            Arial,
            Helvetica,
            sans-serif;
    "
>

    <div
        style="
            max-width:650px;
            margin:30px auto;
            background:#ffffff;
            border-radius:10px;
            overflow:hidden;
            border:1px solid #dddddd;
        "
    >

        <!-- HEADER -->

        <div
            style="
                background:#111827;
                color:#ffffff;
                padding:24px;
            "
        >

            <h1
                style="
                    margin:0;
                    font-size:24px;
                "
            >
                SentinelAI Incident Alert
            </h1>

            <p
                style="
                    margin:8px 0 0 0;
                    color:#d1d5db;
                "
            >
                Automated AI surveillance notification
            </p>

        </div>


        <!-- CONTENT -->

        <div
            style="
                padding:24px;
            "
        >

            <!-- INCIDENT ALERT -->

            <div
                style="
                    background:#fee2e2;
                    border-left:
                        5px solid #dc2626;
                    padding:15px;
                    margin-bottom:20px;
                "
            >

                <strong>
                    {incident_type_html.upper()}
                    DETECTED
                </strong>

                <br><br>

                Threat Level:

                <strong>
                    {threat_level_html}
                </strong>

            </div>


            <!-- INCIDENT DETAILS -->

            <h3>
                Incident Details
            </h3>

            <table
                style="
                    width:100%;
                    border-collapse:
                        collapse;
                    margin-bottom:20px;
                "
            >

                <tr>

                    <td
                        style="
                            padding:8px 0;
                            font-weight:bold;
                        "
                    >
                        Incident ID
                    </td>

                    <td
                        style="
                            padding:8px 0;
                        "
                    >
                        {incident_id_html}
                    </td>

                </tr>


                <tr>

                    <td
                        style="
                            padding:8px 0;
                            font-weight:bold;
                        "
                    >
                        Incident Type
                    </td>

                    <td
                        style="
                            padding:8px 0;
                        "
                    >
                        {incident_type_html}
                    </td>

                </tr>


                <tr>

                    <td
                        style="
                            padding:8px 0;
                            font-weight:bold;
                        "
                    >
                        Threat Level
                    </td>

                    <td
                        style="
                            padding:8px 0;
                        "
                    >
                        {threat_level_html}
                    </td>

                </tr>


                <tr>

                    <td
                        style="
                            padding:8px 0;
                            font-weight:bold;
                        "
                    >
                        Department
                    </td>

                    <td
                        style="
                            padding:8px 0;
                        "
                    >
                        {department_html}
                    </td>

                </tr>


                <tr>

                    <td
                        style="
                            padding:8px 0;
                            font-weight:bold;
                        "
                    >
                        Location
                    </td>

                    <td
                        style="
                            padding:8px 0;
                        "
                    >
                        {location_html}
                    </td>

                </tr>


                <tr>

                    <td
                        style="
                            padding:8px 0;
                            font-weight:bold;
                        "
                    >
                        Detected At
                    </td>

                    <td
                        style="
                            padding:8px 0;
                        "
                    >
                        {date_time_html}
                    </td>

                </tr>

            </table>


            <!-- SUMMARY -->

            <h3>
                Incident Summary
            </h3>

            <p
                style="
                    line-height:1.6;
                "
            >
                {summary_html}
            </p>


            <!-- ACTION -->

            <h3>
                Recommended Action
            </h3>

            <p
                style="
                    line-height:1.6;
                "
            >
                {recommended_action_html}
            </p>


            <!-- RESPONSE -->

            <div
                style="
                    margin-top:25px;
                    padding:15px;
                    background:#f3f4f6;
                    border-radius:6px;
                "
            >

                <strong>
                    SentinelAI Response System
                </strong>

                <p
                    style="
                        margin-bottom:0;
                        line-height:1.5;
                    "
                >
                    This alert was automatically
                    generated after SentinelAI
                    detected a potential anomaly.
                    The incident has been routed
                    to the responsible department.
                </p>

            </div>


            <!-- FOOTER -->

            <hr
                style="
                    border:none;
                    border-top:
                        1px solid #e5e7eb;
                    margin:25px 0;
                "
            >

            <p
                style="
                    color:#6b7280;
                    font-size:13px;
                    line-height:1.5;
                "
            >
                SentinelAI
                AI Surveillance System
                <br>
                Qwen2.5-VL + LoRA
                <br>
                Automated Incident Response
            </p>

        </div>

    </div>

</body>

</html>
"""

    return {
        "subject": subject,
        "html": html,
    }


# ============================================================
# SEND EMAIL
# ============================================================

def send_incident_email(
    incident: dict[str, Any],
) -> dict[str, Any]:
    """
    Send an incident notification using Resend.

    Important:
    The function returns a structured result instead of
    crashing the SentinelAI analysis pipeline if email fails.
    """

    # --------------------------------------------------------
    # Configuration check
    # --------------------------------------------------------

    if not email_configured():

        return {
            "success": False,
            "provider": "resend",
            "channel": "email",
            "status": "NOT_CONFIGURED",
            "message": (
                "RESEND_API_KEY or "
                "ALERT_EMAIL_TO is missing "
                "from the environment."
            ),
        }

    # --------------------------------------------------------
    # Configure Resend
    # --------------------------------------------------------

    resend.api_key = RESEND_API_KEY

    # --------------------------------------------------------
    # Build email
    # --------------------------------------------------------

    email_content = build_incident_email(
        incident
    )

    # --------------------------------------------------------
    # Send
    # --------------------------------------------------------

    try:

        response = resend.Emails.send(
            {
                "from": (
                    "SentinelAI "
                    "<onboarding@resend.dev>"
                ),
                "to": [
                    ALERT_EMAIL_TO
                ],
                "subject": email_content[
                    "subject"
                ],
                "html": email_content[
                    "html"
                ],
            }
        )

        return {
            "success": True,
            "provider": "resend",
            "channel": "email",
            "status": "SENT",
            "recipient": ALERT_EMAIL_TO,
            "response": str(response),
        }

    except Exception as exc:

        return {
            "success": False,
            "provider": "resend",
            "channel": "email",
            "status": "FAILED",
            "recipient": ALERT_EMAIL_TO,
            "error": repr(exc),
        }


# ============================================================
# DISPATCH INCIDENT RESPONSE
# ============================================================

def dispatch_incident_response(
    incident: dict[str, Any],
) -> dict[str, Any]:
    """
    Complete emergency-response dispatch.

    1. Identify incident type.
    2. Route to department.
    3. Set response priority.
    4. Send email.
    5. Return notification status.

    The incident dictionary is updated with routing information.
    """

    incident_type = str(
        incident.get(
            "incident_type",
            "",
        )
    ).strip()

    # --------------------------------------------------------
    # Normal event
    # --------------------------------------------------------

    if incident_type == "Normal":

        incident["department"] = "None"

        incident[
            "response_priority"
        ] = "LOW"

        return {
            "status": "NO_ACTION",
            "department": "None",
            "priority": "LOW",
            "notifications": [],
        }

    # --------------------------------------------------------
    # Route anomaly
    # --------------------------------------------------------

    routing = route_incident(
        incident_type
    )

    department = routing[
        "department"
    ]

    priority = routing[
        "priority"
    ]

    incident[
        "department"
    ] = department

    incident[
        "response_priority"
    ] = priority

    # --------------------------------------------------------
    # Email notification
    # --------------------------------------------------------

    email_result = send_incident_email(
        incident
    )

    # --------------------------------------------------------
    # Final dispatch status
    # --------------------------------------------------------

    if email_result["success"]:

        status = "DISPATCHED"

    else:

        status = "NOTIFICATION_FAILED"

    return {
        "status": status,
        "department": department,
        "priority": priority,
        "notifications": [
            email_result
        ],
    }


# ============================================================
# TEST / DEVELOPMENT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "SENTINELAI - NOTIFICATION SERVICE TEST"
    )
    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print(
        "Email configured:",
        email_configured(),
    )

    print(
        "Recipient:",
        ALERT_EMAIL_TO
        if ALERT_EMAIL_TO
        else "NOT CONFIGURED",
    )

    print()

    # --------------------------------------------------------
    # Routing test
    # --------------------------------------------------------

    print(
        "Routing tests:"
    )

    for incident_type in (
        "Fire",
        "Fight",
        "Road Accident",
        "Normal",
    ):

        routing = route_incident(
            incident_type
        )

        print(
            f"{incident_type:16} -> "
            f"{routing['department']} "
            f"({routing['priority']})"
        )

    print()

    # --------------------------------------------------------
    # Optional real email test
    # --------------------------------------------------------
    #
    # This sends a REAL email.
    #
    # Uncomment the following section only when you want
    # to test the complete notification path.
    # --------------------------------------------------------

    print(
        "Notification service loaded successfully."
    )

    print()

    print("=" * 70)
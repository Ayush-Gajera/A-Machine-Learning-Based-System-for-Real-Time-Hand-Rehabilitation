"""Generate detailed physiotherapist reports from session data"""
import json
from datetime import datetime
from typing import Dict

def generate_physiotherapist_report(session_data: Dict) -> str:
    """
    Generate a detailed report for physiotherapists showing:
    - Session overview
    - Exercise-by-exercise breakdown
    - Joint-level issues and patterns
    - Recovery recommendations
    """
    
    report = []
    report.append("=" * 80)
    report.append("HAND REHABILITATION SESSION REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Session Overview
    report.append("SESSION INFORMATION")
    report.append("-" * 80)
    report.append(f"Session ID: {session_data.get('session_id', 'N/A')}")
    report.append(f"Patient ID: {session_data.get('patient_id', 'default')}")
    report.append(f"Start Time: {session_data.get('start_time', 'N/A')}")
    report.append(f"End Time: {session_data.get('end_time', 'N/A')}")
    report.append(f"Duration: {session_data.get('duration_seconds', 0) / 60:.1f} minutes")
    report.append("")
    
    # Overall Performance
    report.append("OVERALL PERFORMANCE")
    report.append("-" * 80)
    report.append(f"Total Exercise Attempts: {session_data.get('total_attempts', 0)}")
    report.append(f"Successful Completions: {session_data.get('total_completions', 0)}")
    completion_rate = (session_data.get('total_completions', 0) / max(1, session_data.get('total_attempts', 1))) * 100
    report.append(f"Success Rate: {completion_rate:.1f}%")
    report.append(f"Exercises Practiced: {len(session_data.get('exercises', {}))}")
    report.append("")
    
    # Exercise-by-Exercise Analysis
    report.append("DETAILED EXERCISE ANALYSIS")
    report.append("=" * 80)
    
    exercises = session_data.get('exercises', {})
    for ex_name, ex_data in exercises.items():
        report.append("")
        report.append(f"Exercise: {ex_name.replace('-', ' ').replace('_', ' ')}")
        report.append("-" * 80)
        report.append(f"  Attempts: {ex_data.get('total_attempts', 0)}")
        report.append(f"  Completions: {ex_data.get('total_completions', 0)}")
        report.append(f"  Success Rate: {(ex_data.get('total_completions', 0) / max(1, ex_data.get('total_attempts', 1))) * 100:.1f}%")
        report.append(f"  Average Form Quality: {ex_data.get('avg_form_quality', 0):.1f}%")
        report.append(f"  Best Form Quality: {ex_data.get('best_form_quality', 0):.1f}%")
        
        if ex_data.get('avg_completion_time'):
            report.append(f"  Average Completion Time: {ex_data['avg_completion_time']:.1f}s")
        
        # Joint-Level Issues
        joint_issues = ex_data.get('joint_issues', {})
        if joint_issues:
            report.append("")
            report.append("  Joint/Finger Issues Detected:")
            for joint, count in sorted(joint_issues.items(), key=lambda x: x[1], reverse=True):
                frequency = (count / ex_data.get('total_attempts', 1)) * 100
                report.append(f"    - {joint.capitalize()}: {count} times ({frequency:.0f}% of attempts)")
        
        # Attempt Timeline
        attempts = ex_data.get('attempts', [])
        if attempts:
            report.append("")
            report.append("  Attempt Timeline:")
            for i, attempt in enumerate(attempts[:5], 1):  # Show first 5 attempts
                status = "✓ Completed" if attempt.get('completed') else "✗ Incomplete"
                quality = attempt.get('form_quality', 0)
                timestamp = datetime.fromisoformat(attempt['timestamp']).strftime("%H:%M:%S")
                report.append(f"    {i}. [{timestamp}] {status} - Form Quality: {quality:.0f}%")
                
                # Show specific feedback if available
                feedback_details = attempt.get('feedback_details', {})
                if feedback_details.get('bad_fingers'):
                    report.append(f"       Issues: {', '.join(feedback_details['bad_fingers'])}")
            
            if len(attempts) > 5:
                report.append(f"    ... and {len(attempts) - 5} more attempts")
        
        report.append("")
    
    # Clinical Recommendations
    report.append("=" * 80)
    report.append("CLINICAL OBSERVATIONS & RECOMMENDATIONS")
    report.append("=" * 80)
    report.append("")
    
    # Identify problem areas
    all_joint_issues = {}
    for ex_data in exercises.values():
        for joint, count in ex_data.get('joint_issues', {}).items():
            all_joint_issues[joint] = all_joint_issues.get(joint, 0) + count
    
    if all_joint_issues:
        report.append("Most Common Joint/Finger Issues:")
        for joint, count in sorted(all_joint_issues.items(), key=lambda x: x[1], reverse=True)[:3]:
            report.append(f"  - {joint.capitalize()}: {count} total occurrences across all exercises")
        report.append("")
        report.append("Recommendation: Focus additional therapy on the above joints/fingers.")
    else:
        report.append("No significant joint issues detected. Patient showing good form overall.")
    
    report.append("")
    
    # Performance trends
    if completion_rate >= 80:
        report.append("Performance Level: Excellent")
        report.append("  Patient shows strong performance. Consider progressing to more challenging exercises.")
    elif completion_rate >= 60:
        report.append("Performance Level: Good")
        report.append("  Patient is making progress. Continue current exercises with emphasis on form quality.")
    else:
        report.append("Performance Level: Needs Improvement")
        report.append("  Recommend additional sessions focusing on proper form before increasing difficulty.")
    
    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)

def export_report_to_file(session_data: Dict, filename: str = None):
    """Export the report to a text file"""
    if filename is None:
        session_id = session_data.get('session_id', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rehab_report_{session_id[:8]}_{timestamp}.txt"
    
    report_text = generate_physiotherapist_report(session_data)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return filename

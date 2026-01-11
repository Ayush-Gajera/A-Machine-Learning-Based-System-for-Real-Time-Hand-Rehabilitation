"""
Session Manager for Hand Rehabilitation System
Tracks patient progress across multiple sessions
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
import uuid

class SessionManager:
    """Manages rehabilitation sessions and tracks progress over time"""
    
    def __init__(self, storage_file='session_data.json'):
        self.storage_file = storage_file
        self.current_session = None
        self.session_start_time = None
        self._load_data()
    
    def _load_data(self):
        """Load existing session data from file"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    self.data = json.load(f)
            except:
                self.data = {"sessions": []}
        else:
            self.data = {"sessions": []}
    
    def _save_data(self):
        """Save session data to file"""
        with open(self.storage_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def start_session(self, patient_id: Optional[str] = None) -> str:
        """Start a new rehabilitation session"""
        session_id = str(uuid.uuid4())
        self.current_session = {
            "session_id": session_id,
            "patient_id": patient_id or "default",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "exercises": {},
            "total_attempts": 0,
            "total_completions": 0
        }
        self.session_start_time = time.time()
        return session_id
    
    def log_exercise_attempt(self, exercise_name: str, completed: bool, form_quality: float, duration: float = None, detailed_feedback: Dict = None):
        """Log an exercise attempt with detailed metrics and joint-level feedback"""
        if not self.current_session:
            return
        
        if exercise_name not in self.current_session["exercises"]:
            self.current_session["exercises"][exercise_name] = {
                "attempts":[], 
                "total_attempts": 0,
                "total_completions": 0,
                "avg_form_quality": 0.0,
                "avg_completion_time": 0.0,
                "best_form_quality": 0.0,
                "joint_issues": {}  # Track which joints had problems
            }
        
        ex_data = self.current_session["exercises"][exercise_name]
        
        # Record detailed attempt with joint-level feedback
        attempt = {
            "timestamp": datetime.now().isoformat(),
            "completed": completed,
            "form_quality": form_quality,
            "duration": duration,
            "feedback_details": detailed_feedback or {}
        }
        ex_data["attempts"].append(attempt)
        ex_data["total_attempts"] += 1
        
        if completed:
            ex_data["total_completions"] += 1
            self.current_session["total_completions"] += 1
        
        self.current_session["total_attempts"] += 1
        
        # Track joint-specific issues
        if detailed_feedback and "bad_fingers" in detailed_feedback:
            for finger in detailed_feedback["bad_fingers"]:
                if finger not in ex_data["joint_issues"]:
                    ex_data["joint_issues"][finger] = 0
                ex_data["joint_issues"][finger] += 1
        
        # Update averages
        all_qualities = [a["form_quality"] for a in ex_data["attempts"]]
        ex_data["avg_form_quality"] = sum(all_qualities) / len(all_qualities)
        ex_data["best_form_quality"] = max(all_qualities)
        
        if duration is not None:
            completion_times = [a["duration"] for a in ex_data["attempts"] if a["completed"] and a["duration"]]
            if completion_times:
                ex_data["avg_completion_time"] = sum(completion_times) / len(completion_times)
    
    def end_session(self) -> Dict:
        """End current session and return summary"""
        if not self.current_session:
            return {"error": "No active session"}
        
        self.current_session["end_time"] = datetime.now().isoformat()
        self.current_session["duration_seconds"] = time.time() - self.session_start_time
        
        # Calculate summary statistics
        summary = {
            "session_id": self.current_session["session_id"],
            "duration_minutes": round (self.current_session["duration_seconds"] / 60, 2),
            "total_attempts": self.current_session["total_attempts"],
            "total_completions": self.current_session["total_completions"],
            "completion_rate": round(self.current_session["total_completions"] / max(1, self.current_session["total_attempts"]) * 100, 1),
            "exercises_practiced": len(self.current_session["exercises"]),
            "exercise_details": self.current_session["exercises"]
        }
        
        # Save to history
        self.data["sessions"].append(self.current_session)
        self._save_data()
        
        # Reset current session
        self.current_session = None
        self.session_start_time = None
        
        return summary
    
    def get_session_history(self, limit: int = 10) -> List[Dict]:
        """Get recent session history"""
        return self.data["sessions"][-limit:]
    
    def get_exercise_progress(self, exercise_name: str) -> Dict:
        """Get progress for a specific exercise across all sessions"""
        progress = {
            "exercise_name": exercise_name,
            "total_sessions": 0,
            "total_attempts": 0,
            "total_completions": 0,
            "avg_form_quality_trend": [],
            "completion_rate_trend": []
        }
        
        for session in self.data["sessions"]:
            if exercise_name in session["exercises"]:
                ex_data = session["exercises"][exercise_name]
                progress["total_sessions"] += 1
                progress["total_attempts"] += ex_data["total_attempts"]
                progress["total_completions"] += ex_data["total_completions"]
                progress["avg_form_quality_trend"].append({
                    "date": session["start_time"],
                    "quality": ex_data["avg_form_quality"]
                })
                progress["completion_rate_trend"].append({
                    "date": session["start_time"],
                    "rate": round(ex_data["total_completions"] / max(1, ex_data["total_attempts"]) * 100, 1)
                })
        
        return progress
    
    def is_session_active(self) -> bool:
        """Check if there's an active session"""
        return self.current_session is not None

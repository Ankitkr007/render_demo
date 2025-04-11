# agents/email_scheduler.py
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging

# Load environment variables at module level
load_dotenv()

class EmailScheduler:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        
        # Get credentials from environment variables
        self.email = os.getenv("EMAIL_USER")
        self.password = os.getenv("EMAIL_PASSWORD")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def send_interview_invite(self, to_email: str, name: str, position: str):
        """Send interview invitation email"""
        try:
            # Check if credentials are available
            if not self.email or not self.password:
                self.logger.warning("Email credentials not set. Check your .env file")
                return False, "Email credentials not set. Check your .env file"
            
            # Create message
            msg = MIMEText(self.generate_email_content(name, position))
            
            msg["Subject"] = f"Interview Invitation for {position}"
            msg["From"] = self.email
            msg["To"] = to_email
            
            # Send email
            self.logger.info(f"Sending interview invitation to {to_email}")
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email, self.password)
                server.sendmail(self.email, [to_email], msg.as_string())
                
            self.logger.info(f"Email sent successfully to {to_email}")
            return True, "Email sent successfully"
        except Exception as e:
            error_msg = f"Failed to send email: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def generate_email_content(self, name: str, position: str) -> str:
        """Generate the email content with interview slots"""
        return f"""
        Dear {name},
        
        Congratulations! You've been shortlisted for the {position} role.
        
        Available interview slots:
        1. {self._format_date(2)} 10:00 AM - 11:00 AM
        2. {self._format_date(3)} 2:00 PM - 3:00 PM
        3. {self._format_date(4)} 11:00 AM - 12:00 PM
        
        Please reply with your preferred time slot or suggest an alternative time that works better for you.
        
        Best regards,
        Recruitment Team
        """
    
    def _format_date(self, days_add: int) -> str:
        """Format date with weekday name for better readability"""
        future_date = datetime.now() + timedelta(days=days_add)
        return future_date.strftime("%A, %B %d, %Y")
        
    def test_email_connection(self):
        """Test email connection and credentials"""
        try:
            if not self.email or not self.password:
                return "Email credentials not set. Check your .env file"
                
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email, self.password)
                return "Email connection successful"
        except Exception as e:
            return f"Email connection failed: {str(e)}"
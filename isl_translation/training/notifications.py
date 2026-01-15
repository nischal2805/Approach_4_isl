#!/usr/bin/env python3
"""
Training Notifications - Sends progress updates via Telegram or Email.

Usage:
    1. Set environment variables for your notification method
    2. Import and use in trainer.py
"""

import os
import json
import requests
from datetime import datetime
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class TrainingNotifier:
    """Sends training progress notifications via Telegram or email."""
    
    def __init__(self, 
                 telegram_token: Optional[str] = None,
                 telegram_chat_id: Optional[str] = None,
                 email_webhook: Optional[str] = None,
                 notify_every_n_epochs: int = 5):
        """
        Initialize notifier.
        
        Args:
            telegram_token: Telegram bot token (from @BotFather)
            telegram_chat_id: Your Telegram chat ID
            email_webhook: Optional webhook URL for email (e.g., Zapier, IFTTT)
            notify_every_n_epochs: Send notification every N epochs
        """
        # Try environment variables if not provided
        self.telegram_token = telegram_token or os.environ.get('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = telegram_chat_id or os.environ.get('TELEGRAM_CHAT_ID')
        self.email_webhook = email_webhook or os.environ.get('EMAIL_WEBHOOK_URL')
        self.notify_every = notify_every_n_epochs
        
        self.enabled = bool(self.telegram_token and self.telegram_chat_id) or bool(self.email_webhook)
        
        if self.enabled:
            logger.info(f"Notifications enabled (every {notify_every_n_epochs} epochs)")
        else:
            logger.warning("Notifications disabled - set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
    
    def should_notify(self, epoch: int) -> bool:
        """Check if we should send notification for this epoch."""
        return self.enabled and (epoch % self.notify_every == 0 or epoch == 1)
    
    def send_telegram(self, message: str) -> bool:
        """Send message via Telegram bot."""
        if not self.telegram_token or not self.telegram_chat_id:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Telegram notification failed: {e}")
            return False
    
    def send_webhook(self, data: Dict) -> bool:
        """Send data to email webhook."""
        if not self.email_webhook:
            return False
        
        try:
            response = requests.post(self.email_webhook, json=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Webhook notification failed: {e}")
            return False
    
    def notify_epoch(self, epoch: int, total_epochs: int, 
                     train_loss: float, val_loss: float, val_bleu: float,
                     learning_rate: float, time_elapsed: str) -> bool:
        """Send epoch completion notification."""
        if not self.should_notify(epoch):
            return False
        
        # Format message
        progress = epoch / total_epochs * 100
        message = f"""
*ISL Training Progress*

*Epoch {epoch}/{total_epochs}* ({progress:.0f}%)

*Metrics:*
• Train Loss: `{train_loss:.4f}`
• Val Loss: `{val_loss:.4f}`
• Val BLEU: `{val_bleu:.2f}`
• Learning Rate: `{learning_rate:.2e}`

⏱ *Time:* {time_elapsed}

{' Training going well!' if val_bleu > 10 else '⚙️ Model still learning...'}
"""
        
        success = self.send_telegram(message)
        
        # Also try webhook if available
        if self.email_webhook:
            self.send_webhook({
                'epoch': epoch,
                'total_epochs': total_epochs,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_bleu': val_bleu,
                'timestamp': datetime.now().isoformat()
            })
        
        return success
    
    def notify_complete(self, final_metrics: Dict, total_time: str) -> bool:
        """Send training completion notification."""
        if not self.enabled:
            return False
        
        message = f"""
✅ *ISL Training Complete!*

*Final Results:*
• Test Loss: `{final_metrics.get('loss', 'N/A'):.4f}`
• Test BLEU: `{final_metrics.get('bleu', 0):.2f}`

⏱ *Total Time:* {total_time}

Model saved to: `checkpoints/best_model.pt`

Next: Run `python test_inference.py` to test!
"""
        return self.send_telegram(message)
    
    def notify_error(self, error_msg: str, epoch: int) -> bool:
        """Send error notification."""
        if not self.enabled:
            return False
        
        message = f"""
*ISL Training Error!*

*Epoch:* {epoch}
*Error:* `{error_msg[:200]}`

Check `training.log` for details.
"""
        return self.send_telegram(message)


# Simple setup instructions
SETUP_INSTRUCTIONS = """
================================================================================
HOW TO ENABLE TRAINING NOTIFICATIONS
================================================================================

OPTION 1: Telegram (Recommended - Free & Easy)
----------------------------------------------
1. Create a Telegram bot:
   - Message @BotFather on Telegram
   - Send: /newbot
   - Follow prompts, get your BOT_TOKEN

2. Get your Chat ID:
   - Message @userinfobot on Telegram
   - It will reply with your chat ID

3. Set environment variables on server:
   export TELEGRAM_BOT_TOKEN="your_bot_token"
   export TELEGRAM_CHAT_ID="your_chat_id"

4. Run training:
   python training/trainer.py --config configs/train_config.yaml

OPTION 2: Email via Webhook (IFTTT/Zapier)
------------------------------------------
1. Create IFTTT applet or Zapier zap
2. Get webhook URL
3. Set: export EMAIL_WEBHOOK_URL="https://..."

================================================================================
"""

if __name__ == '__main__':
    print(SETUP_INSTRUCTIONS)
    
    # Test notification if credentials provided
    notifier = TrainingNotifier(notify_every_n_epochs=5)
    
    if notifier.enabled:
        print("Testing notification...")
        success = notifier.send_telegram("🧪 Test notification from ISL Training System!")
        print(f"Notification sent: {success}")
    else:
        print("No credentials found. Set environment variables to enable.")

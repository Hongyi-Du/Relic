"""Telegram automation package."""
from tg_automation.analysis import analyze_interactions, UserInteraction
from tg_automation.filters import filter_users, parse_custom_predicate
from tg_automation.engagement import engagement_report, PostEngagement
from tg_automation.growth import create_invite_link, schedule_bulk_invites, trend_dashboard
from tg_automation.export import export_report
from tg_automation.config import AppConfig, AccountConfig, load_config
from tg_automation.client import ClientPool, get_client_pool
from tg_automation.retry import retry_with_backoff, FloodWaitError
__all__ = ['analyze_interactions', 'UserInteraction', 'filter_users', 'parse_custom_predicate', 'engagement_report', 'PostEngagement', 'create_invite_link', 'schedule_bulk_invites', 'trend_dashboard', 'export_report', 'AppConfig', 'AccountConfig', 'load_config', 'ClientPool', 'get_client_pool', 'retry_with_backoff', 'FloodWaitError']

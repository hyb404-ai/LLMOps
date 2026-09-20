"""第三方扩展初始化。"""
from .database_extension import db
from .migrate_extension import migrate

__all__ = ['db','migrate']

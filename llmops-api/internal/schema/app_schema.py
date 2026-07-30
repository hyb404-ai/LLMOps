#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
app_schema - 模块描述

作者: huangyoubin
创建日期: 2026/7/30 19:38
"""
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired, Length


class CompleteForm(FlaskForm):
    """基础聊天接口请求验证"""
    # 必填、长度最大2000
    query = StringField("query", validators=[
        DataRequired(message="用户提问是必填的"),
        Length(max=2000, message="用户提问最大长度2000")
    ])

"""Visualization endpoints that feed the dashboard's Chart.js charts."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session
from ..db import get_db
from ..deps import get_current_user
from ..models import SalesRecord, User
from ..schemas import SalesByCategory, SalesByMonth, SalesPoint
router = APIRouter(prefix='/charts', tags=['charts'])

@router.get('/sales', response_model=list[SalesPoint])
def list_sales(db: Session=Depends(get_db), _: User=Depends(get_current_user)) -> list[SalesRecord]:
    """Return every sales row. Used by the front-end table view."""
    raise NotImplementedError('list_sales is not implemented yet')

@router.get('/sales/by-category', response_model=list[SalesByCategory])
def sales_by_category(db: Session=Depends(get_db), _: User=Depends(get_current_user)) -> list[SalesByCategory]:
    """Return total sales aggregated by category."""
    raise NotImplementedError('sales_by_category is not implemented yet')

@router.get('/sales/by-month', response_model=list[SalesByMonth])
def sales_by_month(db: Session=Depends(get_db), _: User=Depends(get_current_user)) -> list[SalesByMonth]:
    """Return total sales aggregated by month (YYYY-MM)."""
    raise NotImplementedError('sales_by_month is not implemented yet')

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json

class AnalyticsDashboard:
    """
    FastAPI router for analytics dashboard endpoints.
    """
    
    def __init__(self, performance_monitor: 'PerformanceMonitor'):  # Use forward reference
        self.monitor = performance_monitor
        self.router = APIRouter(prefix="/analytics", tags=["Analytics"])
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup all dashboard routes."""
        
        @self.router.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Serve the main analytics dashboard."""
            return self._get_dashboard_html()
        
        @self.router.get("/api/stats")
        async def get_stats():
            """Get overall system statistics."""
            all_stats = self.monitor.get_all_stats()
            
            # Calculate system-wide metrics
            total_requests = sum(stats['total_requests'] for stats in all_stats.values())
            total_successes = sum(stats['total_successes'] for stats in all_stats.values())
            total_failures = sum(stats['total_failures'] for stats in all_stats.values())
            total_cost = sum(stats['total_cost_usd'] for stats in all_stats.values())
            
            return {
                "system_stats": {
                    "total_requests": total_requests,
                    "total_successes": total_successes,
                    "total_failures": total_failures,
                    "success_rate": total_successes / total_requests if total_requests > 0 else 0,
                    "total_cost_usd": total_cost,
                    "active_agents": len(all_stats)
                },
                "agent_stats": all_stats
            }
        
        @self.router.get("/api/metrics")
        async def get_metrics(
            limit: int = Query(100, ge=1, le=1000),
            agent_name: Optional[str] = Query(None),
            hours: Optional[int] = Query(None, ge=1, le=168)  # Max 7 days
        ):
            """Get detailed metrics with optional filtering."""
            
            if hours:
                # Filter by time range
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=hours)
                metrics = self.monitor.get_metrics_by_timerange(start_time, end_time)
            else:
                metrics = self.monitor.get_recent_metrics(limit)
            
            # Filter by agent name if specified
            if agent_name:
                metrics = [m for m in metrics if m['agent_name'] == agent_name]
            
            return {"metrics": metrics}
        
        @self.router.get("/api/agents/{agent_name}/stats")
        async def get_agent_stats(agent_name: str):
            """Get statistics for a specific agent."""
            stats = self.monitor.get_agent_stats(agent_name)
            if not stats:
                raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
            
            return {"agent_name": agent_name, "stats": stats}
        
        @self.router.delete("/api/metrics")
        async def clear_metrics():
            """Clear all metrics (admin only)."""
            self.monitor.clear_metrics()
            return {"message": "All metrics cleared successfully"}
        
        @self.router.get("/api/health")
        async def analytics_health():
            """Analytics system health check."""
            return {
                "status": "healthy",
                "metrics_count": len(self.monitor.metrics_history),
                "active_requests": len(self.monitor.active_requests),
                "tracked_agents": len(self.monitor.agent_stats)
            }
    
    def _get_dashboard_html(self) -> str:
        """Generate the HTML dashboard."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>FluxGraph Analytics Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }
                .header { background: #2c3e50; color: white; padding: 1rem 2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .header h1 { font-size: 1.5rem; font-weight: 600; }
                .container { max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }
                .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
                .stat-card { background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .stat-card h3 { color: #34495e; font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.5px; }
                .stat-card .value { font-size: 2rem; font-weight: bold; color: #2c3e50; }
                .stat-card.success .value { color: #27ae60; }
                .stat-card.error .value { color: #e74c3c; }
                .stat-card.cost .value { color: #f39c12; }
                .charts-container { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 2rem; }
                .chart-card { background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .chart-card h3 { margin-bottom: 1rem; color: #34495e; }
                .agents-table { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; }
                .agents-table h3 { padding: 1.5rem; margin: 0; background: #ecf0f1; color: #2c3e50; }
                .agents-table table { width: 100%; border-collapse: collapse; }
                .agents-table th, .agents-table td { padding: 1rem; text-align: left; border-bottom: 1px solid #ecf0f1; }
                .agents-table th { background: #f8f9fa; font-weight: 600; color: #2c3e50; }
                .loading { text-align: center; padding: 2rem; color: #7f8c8d; }
                .error { background: #e74c3c; color: white; padding: 1rem; border-radius: 4px; margin: 1rem 0; }
                @media (max-width: 768px) {
                    .charts-container { grid-template-columns: 1fr; }
                    .container { padding: 0 0.5rem; }
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 FluxGraph Analytics Dashboard</h1>
            </div>
            
            <div class="container">
                <div id="loading" class="loading">Loading analytics data...</div>
                <div id="error" class="error" style="display: none;"></div>
                <div id="dashboard" style="display: none;">
                    <!-- System Stats -->
                    <div class="stats-grid">
                        <div class="stat-card">
                            <h3>Total Requests</h3>
                            <div class="value" id="total-requests">0</div>
                        </div>
                        <div class="stat-card success">
                            <h3>Success Rate</h3>
                            <div class="value" id="success-rate">0%</div>
                        </div>
                        <div class="stat-card">
                            <h3>Active Agents</h3>
                            <div class="value" id="active-agents">0</div>
                        </div>
                        <div class="stat-card cost">
                            <h3>Total Cost</h3>
                            <div class="value" id="total-cost">$0.00</div>
                        </div>
                    </div>
                    
                    <!-- Charts -->
                    <div class="charts-container">
                        <div class="chart-card">
                            <h3>Request Volume (Last 24h)</h3>
                            <canvas id="requestChart"></canvas>
                        </div>
                        <div class="chart-card">
                            <h3>Average Response Time by Agent</h3>
                            <canvas id="responseTimeChart"></canvas>
                        </div>
                    </div>
                    
                    <!-- Agents Table -->
                    <div class="agents-table">
                        <h3>Agent Performance Summary</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>Agent Name</th>
                                    <th>Requests</th>
                                    <th>Success Rate</th>
                                    <th>Avg Response Time</th>
                                    <th>Total Cost</th>
                                    <th>Last Request</th>
                                </tr>
                            </thead>
                            <tbody id="agents-table-body">
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <script>
                let requestChart, responseTimeChart;
                
                async function loadDashboard() {
                    try {
                        const response = await fetch('/analytics/api/stats');
                        if (!response.ok) throw new Error('Failed to load stats');
                        
                        const data = await response.json();
                        updateSystemStats(data.system_stats);
                        updateAgentsTable(data.agent_stats);
                        await updateCharts();
                        
                        document.getElementById('loading').style.display = 'none';
                        document.getElementById('dashboard').style.display = 'block';
                    } catch (error) {
                        document.getElementById('loading').style.display = 'none';
                        document.getElementById('error').style.display = 'block';
                        document.getElementById('error').textContent = 'Error loading dashboard: ' + error.message;
                    }
                }
                
                function updateSystemStats(stats) {
                    document.getElementById('total-requests').textContent = stats.total_requests.toLocaleString();
                    document.getElementById('success-rate').textContent = (stats.success_rate * 100).toFixed(1) + '%';
                    document.getElementById('active-agents').textContent = stats.active_agents;
                    document.getElementById('total-cost').textContent = '$' + stats.total_cost_usd.toFixed(2);
                }
                
                function updateAgentsTable(agentStats) {
                    const tbody = document.getElementById('agents-table-body');
                    tbody.innerHTML = '';
                    
                    Object.entries(agentStats).forEach(([name, stats]) => {
                        const row = tbody.insertRow();
                        const successRate = stats.total_requests > 0 ? (stats.total_successes / stats.total_requests * 100).toFixed(1) : 0;
                        const lastRequest = stats.last_request ? new Date(stats.last_request).toLocaleString() : 'Never';
                        
                        row.innerHTML = `
                            <td><strong>${name}</strong></td>
                            <td>${stats.total_requests}</td>
                            <td>${successRate}%</td>
                            <td>${stats.avg_duration_ms.toFixed(0)}ms</td>
                            <td>$${stats.total_cost_usd.toFixed(2)}</td>
                            <td>${lastRequest}</td>
                        `;
                    });
                }
                
                async function updateCharts() {
                    try {
                        const response = await fetch('/analytics/api/metrics?hours=24');
                        const data = await response.json();
                        
                        createRequestChart(data.metrics);
                        createResponseTimeChart(data.metrics);
                    } catch (error) {
                        console.error('Error updating charts:', error);
                    }
                }
                
                function createRequestChart(metrics) {
                    const hourlyData = {};
                    const now = new Date();
                    
                    // Initialize 24 hours of data
                    for (let i = 23; i >= 0; i--) {
                        const hour = new Date(now.getTime() - i * 60 * 60 * 1000);
                        const key = hour.toISOString().substring(0, 13); // YYYY-MM-DDTHH
                        hourlyData[key] = 0;
                    }
                    
                    // Count requests per hour
                    metrics.forEach(metric => {
                        const hour = metric.start_time.substring(0, 13);
                        if (hourlyData.hasOwnProperty(hour)) {
                            hourlyData[hour]++;
                        }
                    });
                    
                    const ctx = document.getElementById('requestChart').getContext('2d');
                    if (requestChart) requestChart.destroy();
                    
                    requestChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: Object.keys(hourlyData).map(h => new Date(h).toLocaleTimeString('en-US', {hour: '2-digit'})),
                            datasets: [{
                                label: 'Requests',
                                data: Object.values(hourlyData),
                                borderColor: '#3498db',
                                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                                tension: 0.4
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: { legend: { display: false } },
                            scales: {
                                y: { beginAtZero: true }
                            }
                        }
                    });
                }
                
                function createResponseTimeChart(metrics) {
                    const agentTimes = {};
                    
                    metrics.forEach(metric => {
                        if (metric.duration_ms && metric.success) {
                            if (!agentTimes[metric.agent_name]) {
                                agentTimes[metric.agent_name] = [];
                            }
                            agentTimes[metric.agent_name].push(metric.duration_ms);
                        }
                    });
                    
                    const avgTimes = {};
                    Object.entries(agentTimes).forEach(([agent, times]) => {
                        avgTimes[agent] = times.reduce((a, b) => a + b, 0) / times.length;
                    });
                    
                    const ctx = document.getElementById('responseTimeChart').getContext('2d');
                    if (responseTimeChart) responseTimeChart.destroy();
                    
                    responseTimeChart = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: Object.keys(avgTimes),
                            datasets: [{
                                label: 'Avg Response Time (ms)',
                                data: Object.values(avgTimes),
                                backgroundColor: '#e74c3c'
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: { legend: { display: false } },
                            scales: {
                                y: { beginAtZero: true }
                            }
                        }
                    });
                }
                
                // Auto-refresh dashboard every 30 seconds
                setInterval(loadDashboard, 30000);
                
                // Load dashboard on page load
                loadDashboard();
            </script>
        </body>
        </html>
        """
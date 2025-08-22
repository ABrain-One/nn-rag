#!/usr/bin/env python3
"""
Smart RAG-NN Pipeline - Integrates smart dependency resolution with health checking
Implements user's suggested strategy without hardcoding
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os

from ..utils.dependency_resolver import SmartDependencyResolver
from .health_checker import HealthChecker

log = logging.getLogger(__name__)

class SmartRAGPipeline:
    """Smart RAG-NN Pipeline with intelligent code discovery and dependency resolution"""
    
    def __init__(self, github_token: str, output_dir: str = "output", blocks_dir: str = "blocks"):
        self.github_token = github_token
        self.output_dir = Path(output_dir)
        self.blocks_dir = Path(blocks_dir)
        self.resolver = SmartDependencyResolver(github_token)
        self.health_checker = HealthChecker(str(self.output_dir), str(self.blocks_dir))
        
        # Ensure directories exist
        self.output_dir.mkdir(exist_ok=True)
        self.blocks_dir.mkdir(exist_ok=True)
        
        # Pipeline statistics
        self.stats = {
            "total_blocks": 0,
            "successful_blocks": 0,
            "failed_blocks": 0,
            "healthy_blocks": 0,
            "unhealthy_blocks": 0,
            "total_time": 0,
            "average_time_per_block": 0
        }
    
    async def process_blocks(self, block_names: List[str]) -> Dict:
        """
        Process multiple blocks using the smart strategy
        
        Args:
            block_names: List of block names to process
            
        Returns:
            Dict with processing results and statistics
        """
        start_time = time.time()
        
        log.info(f"🚀 Starting Smart RAG-NN Pipeline for {len(block_names)} blocks")
        
        self.stats["total_blocks"] = len(block_names)
        
        # Process each block
        results = []
        for block_name in block_names:
            log.info(f"🔍 Processing block: {block_name}")
            
            result = await self._process_single_block(block_name)
            results.append(result)
            
            if result["success"]:
                self.stats["successful_blocks"] += 1
            else:
                self.stats["failed_blocks"] += 1
        
        # Save all processed blocks to output directory
        await self._save_blocks_to_output(results)
        
        # Run comprehensive health check
        health_results = await self._run_health_check()
        
        # Calculate final statistics
        self.stats["total_time"] = time.time() - start_time
        self.stats["average_time_per_block"] = self.stats["total_time"] / max(1, len(block_names))
        
        return {
            "results": results,
            "health_results": health_results,
            "statistics": self.stats
        }
    
    async def _process_single_block(self, block_name: str) -> Dict:
        """Process a single block using smart resolution"""
        try:
            log.info(f"🔍 Smart processing: {block_name}")
            
            # Use smart dependency resolver
            success, code, metadata = await self.resolver.process_block_complete(block_name)
            
            if success:
                result = {
                    "block_name": block_name,
                    "success": True,
                    "repository": metadata["repository"],
                    "path": metadata["path"],
                    "code": code,
                    "metadata": metadata
                }
                
                log.info(f"✅ Successfully processed {block_name}")
                log.info(f"   📍 Repository: {metadata['repository']}")
                log.info(f"   📁 Path: {metadata['path']}")
                log.info(f"   🔗 Dependencies resolved: {metadata['dependencies_resolved']}")
                log.info(f"   📏 Lines: {metadata['line_count']}")
                log.info(f"   🚀 PyTorch: {metadata['has_pytorch']}")
                log.info(f"   📊 Score: {metadata['score']:.2f}")
                
                return result
            else:
                log.error(f"❌ Failed to process {block_name}: {metadata.get('error', 'Unknown error')}")
                return {
                    "block_name": block_name,
                    "success": False,
                    "error": metadata.get('error', 'Unknown error')
                }
                
        except Exception as e:
            log.error(f"❌ Error processing {block_name}: {e}")
            return {
                "block_name": block_name,
                "success": False,
                "error": str(e)
            }
    
    async def _save_blocks_to_output(self, results: List[Dict]) -> None:
        """Save all processed blocks to output directory"""
        for result in results:
            if result["success"]:
                block_name = result["block_name"]
                code = result["code"]
                
                output_file = self.output_dir / f"{block_name}.py"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                log.info(f"💾 Saved {block_name} to output directory")
    
    async def _run_health_check(self) -> Dict:
        """Run comprehensive health check on all processed files"""
        log.info("🔍 Running comprehensive health check...")
        
        # Find all Python files in output directory
        python_files = list(self.output_dir.glob("*.py"))
        
        if not python_files:
            log.warning("No Python files found in output directory")
            return {"healthy_files": [], "unhealthy_files": []}
        
        # Check health of each file
        health_results = {}
        for file_path in python_files:
            is_healthy, details = self.health_checker.check_file_health(file_path)
            health_results[file_path.name] = (is_healthy, details)
            
            if is_healthy:
                log.info(f"✅ {file_path.name}: Healthy")
                self.stats["healthy_blocks"] += 1
            else:
                log.info(f"❌ {file_path.name}: Unhealthy")
                self.stats["unhealthy_blocks"] += 1
        
        # Move healthy files to blocks directory
        healthy_count, unhealthy_count = self.health_checker.move_healthy_files(health_results)
        
        log.info(f"📁 Organization complete:")
        log.info(f"   ✅ Moved {healthy_count} healthy files to blocks/")
        log.info(f"   ❌ Kept {unhealthy_count} unhealthy files in output/")
        
        return health_results
    
    def generate_report(self, pipeline_results: Dict) -> str:
        """Generate a comprehensive pipeline report"""
        results = pipeline_results["results"]
        health_results = pipeline_results["health_results"]
        stats = pipeline_results["statistics"]
        
        report_lines = []
        
        # Header
        report_lines.append("🚀 SMART RAG-NN PIPELINE REPORT")
        report_lines.append("=" * 60)
        
        # Statistics
        report_lines.append(f"📊 PIPELINE STATISTICS:")
        report_lines.append(f"   Total blocks processed: {stats['total_blocks']}")
        report_lines.append(f"   ✅ Successful: {stats['successful_blocks']}")
        report_lines.append(f"   ❌ Failed: {stats['failed_blocks']}")
        report_lines.append(f"   🏥 Healthy: {stats['healthy_blocks']}")
        report_lines.append(f"   🚑 Unhealthy: {stats['unhealthy_blocks']}")
        report_lines.append(f"   ⏱️  Total time: {stats['total_time']:.2f}s")
        report_lines.append(f"   📈 Avg time per block: {stats['average_time_per_block']:.2f}s")
        
        health_rate = (stats['healthy_blocks'] / max(1, stats['total_blocks'])) * 100
        report_lines.append(f"   💊 Health rate: {health_rate:.1f}%")
        report_lines.append("")
        
        # Individual block results
        report_lines.append("📋 BLOCK PROCESSING RESULTS:")
        report_lines.append("-" * 40)
        
        for result in results:
            if result["success"]:
                metadata = result["metadata"]
                report_lines.append(f"✅ {result['block_name']}")
                report_lines.append(f"   📍 Repository: {metadata['repository']}")
                report_lines.append(f"   📁 Path: {metadata['path']}")
                report_lines.append(f"   🔗 Dependencies: {metadata['dependencies_resolved']}/{metadata['total_dependencies']}")
                report_lines.append(f"   📏 Lines: {metadata['line_count']}")
                report_lines.append(f"   🚀 PyTorch: {'Yes' if metadata['has_pytorch'] else 'No'}")
                report_lines.append(f"   📊 Score: {metadata['score']:.2f}")
            else:
                report_lines.append(f"❌ {result['block_name']}")
                report_lines.append(f"   Error: {result['error']}")
            report_lines.append("")
        
        # Health check summary
        if health_results:
            report_lines.append("🏥 HEALTH CHECK SUMMARY:")
            report_lines.append("-" * 40)
            
            for filename, (is_healthy, details) in health_results.items():
                status = "✅ Healthy" if is_healthy else "❌ Unhealthy"
                report_lines.append(f"{status}: {filename}")
                
                if not is_healthy:
                    # Show key issues
                    for check, result in details.items():
                        if check.endswith('_ok') and result is False:
                            issue_key = check.replace('_ok', '')
                            if issue_key in details:
                                report_lines.append(f"   ⚠️  {issue_key}: {details[issue_key]}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("💡 RECOMMENDATIONS:")
        report_lines.append("-" * 40)
        
        if health_rate >= 75:
            report_lines.append("🎉 Excellent! Most blocks are healthy and ready to use.")
        elif health_rate >= 50:
            report_lines.append("👍 Good progress. Consider fixing remaining unhealthy blocks.")
        elif health_rate >= 25:
            report_lines.append("⚠️  Some blocks need attention. Check dependency resolution.")
        else:
            report_lines.append("🔧 Many blocks need fixing. Review search strategy and dependencies.")
        
        return "\n".join(report_lines)
    
    async def save_pipeline_summary(self, pipeline_results: Dict, summary_file: str = "pipeline_summary.json") -> None:
        """Save pipeline summary to JSON file"""
        import json
        
        # Prepare summary data (without large code content)
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "statistics": pipeline_results["statistics"],
            "blocks": []
        }
        
        for result in pipeline_results["results"]:
            if result["success"]:
                metadata = result["metadata"]
                summary["blocks"].append({
                    "name": result["block_name"],
                    "success": True,
                    "repository": metadata["repository"],
                    "path": metadata["path"],
                    "dependencies_resolved": metadata["dependencies_resolved"],
                    "total_dependencies": metadata["total_dependencies"],
                    "line_count": metadata["line_count"],
                    "has_pytorch": metadata["has_pytorch"],
                    "score": metadata["score"]
                })
            else:
                summary["blocks"].append({
                    "name": result["block_name"],
                    "success": False,
                    "error": result["error"]
                })
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        log.info(f"💾 Saved pipeline summary to {summary_file}")

async def run_smart_pipeline(block_names: List[str], github_token: str, 
                            output_dir: str = "output", blocks_dir: str = "blocks") -> Dict:
    """
    Run the complete smart RAG-NN pipeline
    
    Args:
        block_names: List of block names to process
        github_token: GitHub authentication token
        output_dir: Directory for temporary output files
        blocks_dir: Directory for healthy/final blocks
        
    Returns:
        Complete pipeline results
    """
    pipeline = SmartRAGPipeline(github_token, output_dir, blocks_dir)
    
    # Process all blocks
    results = await pipeline.process_blocks(block_names)
    
    # Generate and display report
    report = pipeline.generate_report(results)
    print(report)
    
    # Save summary
    await pipeline.save_pipeline_summary(results)
    
    return results

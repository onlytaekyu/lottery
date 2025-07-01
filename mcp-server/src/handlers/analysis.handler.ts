import * as path from 'node:path';
import { execPython } from '../utils/python-executor.js';

export async function runDataAnalysis(projectRoot: string, args: any) {
  const script = path.join(projectRoot, 'src', 'run', 'run1_00_data_analysis.py');
  return execPythonSafe(projectRoot, script, args);
}

export async function generateFeatureVectors(projectRoot: string, args: any) {
  const script = path.join(projectRoot, 'src', 'analysis', 'feature_extractor.py');
  return execPythonSafe(projectRoot, script, args);
}

export async function analyzePatterns(projectRoot: string, args: any) {
  const script = path.join(projectRoot, 'src', 'analysis', 'pattern_analyzer.py');
  return execPythonSafe(projectRoot, script, args);
}

async function execPythonSafe(projectRoot: string, script: string, args: any) {
  if (!validateScript(projectRoot, script)) throw new Error('Invalid script path');
  const sanitizedArgs = sanitizeArgs(args);
  return await execPython(projectRoot, script, sanitizedArgs);
}

function validateScript(projectRoot: string, scriptPath: string) {
  const normalized = path.normalize(scriptPath);
  return normalized.startsWith(projectRoot);
}

function sanitizeArgs(args: any): string[] {
  return (Array.isArray(args) ? args : []).map((a) => String(a).replace(/[;&|`$]/g, ''));
} 
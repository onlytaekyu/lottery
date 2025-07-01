/// <reference types="node" />
import { ChildProcessWithoutNullStreams, spawn } from 'child_process';
import * as fs from 'fs-extra';
import * as path from 'path';

export interface ExecResult {
  stdout: string;
  stderr: string;
}

export async function execPython(
  projectRoot: string,
  script: string,
  args: string[] = [],
  timeoutMs = 60_000
): Promise<ExecResult> {
  return new Promise((resolve, reject) => {
    const venvPath = path.join(projectRoot, '.venv');
    const pythonBin = process.platform === 'win32'
      ? path.join(venvPath, 'Scripts', 'python.exe')
      : path.join(venvPath, 'bin', 'python');

    const pythonCmd = fs.existsSync(pythonBin) ? pythonBin : 'python';

    const proc: ChildProcessWithoutNullStreams = spawn(pythonCmd, [script, ...args], {
      cwd: projectRoot,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    const timer = setTimeout(() => {
      proc.kill('SIGKILL');
      reject(new Error('Python process timeout'));
    }, timeoutMs);

    proc.stdout.on('data', (d: Buffer) => (stdout += d.toString()));
    proc.stderr.on('data', (d: Buffer) => (stderr += d.toString()));

    proc.on('close', (code: number | null) => {
      clearTimeout(timer);
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error(`Python exited with code ${code}: ${stderr}`));
    });

    proc.on('error', reject);
  });
} 
// ============================================================================
// MCP 서버 구현 - 로또 AI 시스템 통합 (수정된 최신 버전)
// ============================================================================

import { McpServer, ResourceTemplate } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { spawn } from 'child_process';
import * as fs from 'fs-extra';
import * as yaml from 'js-yaml';
import * as path from 'path';
import { z } from 'zod';

interface PythonResult {
  stdout: string;
  stderr: string;
}

class LotteryAIMCPServer {
  private server: McpServer;
  private projectRoot: string;

  constructor() {
    // McpServer 생성 (최신 API)
    this.server = new McpServer({
      name: "lottery-ai-server",
      version: "1.0.0",
    });
    
    this.projectRoot = path.resolve(process.cwd());
    this.setupTools();
    this.setupResources();
  }

  // ============================================================================
  // 도구 등록 (최신 registerTool 방식)
  // ============================================================================
  
  private setupTools() {
    // 1. 데이터 분석 도구
    this.server.registerTool(
      "run_data_analysis",
      {
        title: "로또 데이터 분석 실행",
        description: "과거 로또 당첨 번호를 분석하여 패턴을 추출합니다",
        inputSchema: {
          rounds: z.number().optional().describe("분석할 회차 수 (기본값: 전체)")
        }
      },
      async ({ rounds }: { rounds?: number }) => {
        return await this.runDataAnalysis(rounds);
      }
    );

    // 2. LightGBM 모델 학습
    this.server.registerTool(
      "train_lightgbm",
      {
        title: "LightGBM 모델 학습",
        description: "LightGBM 알고리즘을 사용하여 로또 예측 모델을 학습합니다",
        inputSchema: {
          epochs: z.number().optional().describe("학습 에포크 수 (기본값: 100)")
        }
      },
      async ({ epochs }: { epochs?: number }) => {
        return await this.trainLightGBM(epochs);
      }
    );

    // 3. 예측 생성
    this.server.registerTool(
      "generate_predictions",
      {
        title: "로또 번호 예측",
        description: "학습된 모델을 사용하여 다음 회차 로또 번호를 예측합니다",
        inputSchema: {
          count: z.number().optional().describe("생성할 예측 조합 수 (기본값: 5)")
        }
      },
      async ({ count }: { count?: number }) => {
        return await this.generatePredictions(count);
      }
    );

    // 4. 시스템 상태 확인
    this.server.registerTool(
      "get_system_status",
      {
        title: "시스템 상태 확인",
        description: "로또 AI 시스템의 현재 상태를 확인합니다",
        inputSchema: {}
      },
      async () => {
        return await this.getSystemStatus();
      }
    );

    // 5. 성능 메트릭 조회
    this.server.registerTool(
      "get_performance_metrics",
      {
        title: "성능 메트릭 조회",
        description: "모델의 성능 지표를 조회합니다",
        inputSchema: {
          model_name: z.string().optional().describe("조회할 모델명 (기본값: 모든 모델)")
        }
      },
      async ({ model_name }: { model_name?: string }) => {
        return await this.getPerformanceMetrics(model_name);
      }
    );

    // 6. 설정 업데이트
    this.server.registerTool(
      "update_config",
      {
        title: "설정 업데이트",
        description: "시스템 설정을 업데이트합니다",
        inputSchema: {
          config_path: z.string().describe("설정 파일 경로"),
          updates: z.record(z.any()).describe("업데이트할 설정 값들")
        }
      },
      async ({ config_path, updates }: { config_path: string; updates: Record<string, any> }) => {
        return await this.updateConfig(config_path, updates);
      }
    );
  }

  // ============================================================================
  // 리소스 등록 (최신 registerResource 방식)
  // ============================================================================
  
  private setupResources() {
    // 1. 최신 분석 결과
    this.server.registerResource(
      "analysis-latest",
      new ResourceTemplate("lottery://analysis/latest", { list: undefined }),
      {
        title: "최신 분석 결과",
        description: "가장 최근의 데이터 분석 결과를 제공합니다"
      },
      async (uri) => {
        const analysisFile = path.join(this.projectRoot, 'data/result/latest_analysis.json');
        
        try {
          const content = await fs.readFile(analysisFile, 'utf8');
          return {
            contents: [{
              uri: uri.href,
              text: content,
              mimeType: "application/json"
            }]
          };
        } catch (error) {
          return {
            contents: [{
              uri: uri.href,
              text: JSON.stringify({ error: "분석 결과를 찾을 수 없습니다" }),
              mimeType: "application/json"
            }]
          };
        }
      }
    );

    // 2. 모델 성능 지표
    this.server.registerResource(
      "models-performance",
      new ResourceTemplate("lottery://models/performance", { list: undefined }),
      {
        title: "모델 성능 지표",
        description: "학습된 모델들의 성능 지표를 제공합니다"
      },
      async (uri) => {
        const performanceFile = path.join(this.projectRoot, 'data/result/model_performance.json');
        
        try {
          const content = await fs.readFile(performanceFile, 'utf8');
          return {
            contents: [{
              uri: uri.href,
              text: content,
              mimeType: "application/json"
            }]
          };
        } catch (error) {
          return {
            contents: [{
              uri: uri.href,
              text: JSON.stringify({ error: "성능 지표를 찾을 수 없습니다" }),
              mimeType: "application/json"
            }]
          };
        }
      }
    );
  }

  // ============================================================================
  // 실제 구현 메서드들
  // ============================================================================

  private async runDataAnalysis(rounds?: number): Promise<{ content: Array<{ type: string; text: string }> }> {
    try {
      const scriptPath = path.join(this.projectRoot, 'src/run/run1_00_data_analysis.py');
      const args = rounds ? ['--rounds', rounds.toString()] : [];
      const result = await this.executePythonScript(scriptPath, args);
      
      return {
        content: [
          {
            type: "text",
            text: `✅ 데이터 분석 완료:\n\n${result.stdout}\n\n${result.stderr ? `⚠️ 경고: ${result.stderr}` : ''}`
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text", 
            text: `❌ 분석 실행 중 오류 발생: ${error instanceof Error ? error.message : String(error)}`
          }
        ]
      };
    }
  }

  private async trainLightGBM(epochs?: number): Promise<{ content: Array<{ type: string; text: string }> }> {
    try {
      const scriptPath = path.join(this.projectRoot, 'src/pipeline/train_pipeline.py');
      const args = ['--model', 'lightgbm'];
      if (epochs) {
        args.push('--epochs', epochs.toString());
      }
      
      const result = await this.executePythonScript(scriptPath, args);
      
      return {
        content: [
          {
            type: "text",
            text: `✅ LightGBM 모델 학습 완료:\n\n${result.stdout}`
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ 모델 학습 중 오류 발생: ${error instanceof Error ? error.message : String(error)}`
          }
        ]
      };
    }
  }

  private async generatePredictions(count?: number): Promise<{ content: Array<{ type: string; text: string }> }> {
    try {
      const scriptPath = path.join(this.projectRoot, 'src/run/run4_00_batch_prediction.py');
      const args = count ? ['--count', count.toString()] : [];
      const result = await this.executePythonScript(scriptPath, args);
      
      return {
        content: [
          {
            type: "text",
            text: `🎯 로또 번호 예측 완료:\n\n${result.stdout}`
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ 예측 생성 중 오류 발생: ${error instanceof Error ? error.message : String(error)}`
          }
        ]
      };
    }
  }

  private async getSystemStatus(): Promise<{ content: Array<{ type: string; text: string }> }> {
    try {
      const status = {
        timestamp: new Date().toISOString(),
        system: {
          memory: process.memoryUsage(),
          uptime: process.uptime(),
          platform: process.platform,
          nodeVersion: process.version
        },
        analysis: await this.checkAnalysisStatus(),
        models: await this.checkModelsStatus(),
        cache: await this.checkCacheStatus()
      };

      return {
        content: [
          {
            type: "text",
            text: `📊 시스템 상태 보고서:\n\n${JSON.stringify(status, null, 2)}`
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ 상태 확인 중 오류: ${error instanceof Error ? error.message : String(error)}`
          }
        ]
      };
    }
  }

  private async getPerformanceMetrics(modelName?: string): Promise<{ content: Array<{ type: string; text: string }> }> {
    try {
      const metricsFile = path.join(this.projectRoot, 'data/result/performance_reports');
      
      if (!await fs.pathExists(metricsFile)) {
        return {
          content: [
            {
              type: "text",
              text: "📈 성능 메트릭 파일이 존재하지 않습니다. 먼저 모델을 학습해주세요."
            }
          ]
        };
      }

      const files = await fs.readdir(metricsFile);
      const relevantFiles = modelName 
        ? files.filter(f => f.includes(modelName))
        : files;

      let metricsData = '';
      for (const file of relevantFiles.slice(0, 5)) { // 최대 5개 파일
        const filePath = path.join(metricsFile, file);
        const content = await fs.readFile(filePath, 'utf8');
        metricsData += `\n=== ${file} ===\n${content}\n`;
      }

      return {
        content: [
          {
            type: "text",
            text: `📊 성능 메트릭 보고서:${metricsData}`
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ 메트릭 조회 중 오류: ${error instanceof Error ? error.message : String(error)}`
          }
        ]
      };
    }
  }

  private async updateConfig(configPath: string, updates: Record<string, any>): Promise<{ content: Array<{ type: string; text: string }> }> {
    try {
      const fullPath = path.join(this.projectRoot, configPath);
      
      if (!this.validatePath(fullPath)) {
        throw new Error("잘못된 파일 경로입니다");
      }

      const currentConfig = yaml.load(await fs.readFile(fullPath, 'utf8')) as Record<string, any>;
      const updatedConfig = { ...currentConfig, ...updates };
      
      await fs.writeFile(fullPath, yaml.dump(updatedConfig), 'utf8');

      return {
        content: [
          {
            type: "text",
            text: `✅ 설정 업데이트 완료: ${configPath}\n\n업데이트된 설정:\n${JSON.stringify(updates, null, 2)}`
          }
        ]
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `❌ 설정 업데이트 실패: ${error instanceof Error ? error.message : String(error)}`
          }
        ]
      };
    }
  }

  // ============================================================================
  // 유틸리티 메서드들
  // ============================================================================

  private getVenvCommand(): string {
    const venvPath = path.join(this.projectRoot, 'venv', 'Scripts', 'python.exe');
    return fs.existsSync(venvPath) ? venvPath : 'python';
  }

  private validatePath(filePath: string): boolean {
    const normalizedPath = path.normalize(filePath);
    return normalizedPath.startsWith(this.projectRoot);
  }

  private sanitizeArgs(args: any[]): string[] {
    return args.map(arg => String(arg).replace(/[;&|`$]/g, ''));
  }

  private async executePythonScript(
    scriptPath: string, 
    args: any[] = [],
    timeout: number = 300000 // 5분 타임아웃
  ): Promise<PythonResult> {
    if (!this.validatePath(scriptPath)) {
      throw new Error("잘못된 스크립트 경로입니다");
    }

    const sanitizedArgs = this.sanitizeArgs(args);
    const pythonCmd = this.getVenvCommand();

    return new Promise((resolve, reject) => {
      const python = spawn(pythonCmd, [scriptPath, ...sanitizedArgs], {
        cwd: this.projectRoot,
        stdio: ['pipe', 'pipe', 'pipe'],
        timeout
      });

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      python.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      python.on('close', (code) => {
        if (code === 0) {
          resolve({ stdout, stderr });
        } else {
          reject(new Error(`Python script exited with code ${code}\nStderr: ${stderr}`));
        }
      });

      python.on('error', (error) => {
        reject(error);
      });

      // 타임아웃 처리
      setTimeout(() => {
        python.kill('SIGTERM');
        reject(new Error(`Python script timed out after ${timeout}ms`));
      }, timeout);
    });
  }

  private async checkAnalysisStatus() {
    const cacheDir = path.join(this.projectRoot, 'data/cache');
    const vectorFile = path.join(cacheDir, 'feature_vector_full.npy');
    
    return {
      feature_vectors_exist: await fs.pathExists(vectorFile),
      last_analysis: await this.getLastModified(vectorFile),
      cache_size: await this.getDirSize(cacheDir)
    };
  }

  private async checkModelsStatus() {
    const modelsDir = path.join(this.projectRoot, 'savedModels');
    
    return {
      models_directory_exists: await fs.pathExists(modelsDir),
      available_models: await fs.pathExists(modelsDir) ? await fs.readdir(modelsDir) : [],
      last_training: await this.getLastModified(modelsDir)
    };
  }

  private async checkCacheStatus() {
    const cacheDir = path.join(this.projectRoot, 'data/cache');
    
    return {
      cache_directory_exists: await fs.pathExists(cacheDir),
      cache_files: await fs.pathExists(cacheDir) ? (await fs.readdir(cacheDir)).length : 0,
      total_cache_size: await this.getDirSize(cacheDir)
    };
  }

  private async getLastModified(filePath: string): Promise<string | null> {
    try {
      const stats = await fs.stat(filePath);
      return stats.mtime.toISOString();
    } catch {
      return null;
    }
  }

  private async getDirSize(dirPath: string): Promise<number> {
    try {
      let totalSize = 0;
      const files = await fs.readdir(dirPath);
      
      for (const file of files) {
        const filePath = path.join(dirPath, file);
        const stats = await fs.stat(filePath);
        if (stats.isFile()) {
          totalSize += stats.size;
        }
      }
      
      return totalSize;
    } catch {
      return 0;
    }
  }

  // ============================================================================
  // 서버 시작
  // ============================================================================
  
  async start() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.log('🚀 Lottery AI MCP Server started successfully!');
  }
}

// 서버 시작
async function main() {
  try {
    const server = new LotteryAIMCPServer();
    await server.start();
  } catch (error) {
    console.error('❌ Failed to start MCP server:', error);
    process.exit(1);
  }
}

main();
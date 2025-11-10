import { NextResponse } from 'next/server';
import { WalletManager } from '@/lib/wallet/manager';
import fs from 'fs';
import path from 'path';

const SECURE_DIR = path.join(process.cwd(), 'secure');
const MFA_SECRET_FILE = path.join(SECURE_DIR, 'mfa-secret.json.enc');

export async function GET() {
  try {
    const manager = new WalletManager();

    const mfaEnabled = fs.existsSync(MFA_SECRET_FILE);

    return NextResponse.json({
      exists: manager.walletExists(),
      mfaEnabled,
    });
  } catch (error: any) {
    console.error('[API] Wallet status error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to check wallet status' },
      { status: 500 }
    );
  }
}

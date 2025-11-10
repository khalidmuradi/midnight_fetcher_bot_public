# MFA Implementation TODO

## 1. Add speakeasy dependency
- [x] Update package.json to include speakeasy for TOTP MFA
- [x] Run npm install to install the new dependency

## 2. Extend encryption.ts for MFA secrets
- [x] Add functions to encrypt/decrypt MFA secrets using existing AES-256-GCM
- [x] Ensure MFA secrets are stored securely alongside wallet data

## 3. Modify WalletManager for MFA support
- [x] Add MFA secret generation during wallet creation
- [x] Add MFA verification during wallet loading
- [x] Store MFA secret encrypted in secure directory
- [x] Handle backward compatibility for existing wallets

## 4. Update wallet create API
- [x] Generate MFA secret during wallet creation
- [x] Return MFA setup information (secret, QR code URL)
- [x] Log MFA setup events

## 5. Update wallet load API
- [x] Require MFA code in request body
- [x] Verify MFA code against stored secret
- [x] Log MFA verification attempts (success/failure)

## 6. Enhance logger for MFA events
- [x] Add 'mfa' category to logger
- [x] Log MFA setup, verification attempts, and failures

## 7. Update wallet status API
- [x] Include MFA enabled status in response
- [x] Show if MFA is configured for the wallet

## Testing and Validation
- [ ] Test MFA setup flow end-to-end
- [ ] Test MFA verification during wallet loading
- [ ] Ensure backward compatibility for existing wallets
- [ ] Test error handling for invalid MFA codes

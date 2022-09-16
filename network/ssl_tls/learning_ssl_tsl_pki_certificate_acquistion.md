# PKI Overview

## Source

https://www.linkedin.com/learning/learning-ssl-tls/

## SSL vs. TLS

SSL and TLS are protocols that are used to secure network communications

- Uses PKI certificates  (also called X.509 certificate) and related keys to secure network communication

- Encryption (confidentiality)

- Digital signatures (authentication, non-repudiation): determine whether or not a message is authentic and it came from who it says it came from

- Hashing (integrity): integrity of files through file hashing

- Both are application-specific (it must be configured separately for HTTP, SMTP, and so on)

## Secure Sockets Layer

- SSL

- Developed be Netscape in the early 1990s

- SSL v1-v3; non of these should be used unless required for legacy interoperability

- Superseded by TLS

## Transport Security Layer

- TLS

- Introduced in 1999

- Replaces SSL

- TLS v1.0-1.3 (v3 as of August 2018)

- Disable TLS v1.0; this is required to comply with PCI DSS

## SSL (TLS) VPN

- firewall friendly

![ssl_vpn!](./img/ssl_vpn.png)

Check TLS Support on Linux

```bash
openssl ciphers -v
```

## Acquire a web server certificate using OpenSSL

Generate private key (better to have password)

```bash
# Linux
openssl genrsa -aes256 --out www.fakesitelocal.key 2048

# Mac
openssl genrsa -aes256 -out www.fakesitelocal.key 2048
```

Generate a certificate signing request 

```bash
# Mac (no otherinfo.ext output)
openssl req -new -key www.fakesitelocal.key -out www.fake.site.local.csr
```

Generate web server certificate

```bash
# Lunix (with otherinfo.ext)
openssl x509 -req -in www.fake.site.local.csr -CA FakeDomain2CA.pem -CAkey CAprivate.key -CAcreateserial -extfile otherinfo.ext -out www.fakesite.kocal.crt -days 365 -sha256

# Mac (without otherinfo.ext)
openssl x509 -req -in www.fake.site.local.csr -CA FakeDomain2CA.pem -CAkey CAprivate.key -CAcreateserial -out www.fakesite.kocal.crt -days 365 -sha256
```

Check

```bash
cat /etc/apache2/sites-enabled/000-default.conf
```

Restart server

```bash
service apache2 restart
```

Check status

```bash
service apache2 status
```

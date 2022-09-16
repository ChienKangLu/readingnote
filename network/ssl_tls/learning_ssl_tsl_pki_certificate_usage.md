# PKI CA Usage

## Source

[Learning SSL/TLS Online Class | LinkedIn Learning, formerly Lynda.com](https://www.linkedin.com/learning/learning-ssl-tls/)

## Hashing and digital signature

### Hashing

- Used to verify the integrity of network messages, files, and machine boot-up settings (make sure that these things have not been tampered with, they've not been modified by unauthorized parties)

- Hashing then doesn't provide data confidentiality

- Hashing can also be used with SSL and TLS

- Hashing uses a one-way algorithm that results in a unique value and that unique value is called a hash or a message digest. And it's unique based on the data itself that was hashed.

### Common Hashing algorithm

- SHA-1

- SHA-2

- SHA-3

- MD5

- RIPEMD

If you're doing that within your organization, you would get to choose the hashing algorithm used by that authority to digitally sign certificates.

### Digital Signatures

- Provides data authentication, integrity, and non-repudiation

- We need to make sure that we trust that the message came from who it says it came from. How can you do that? 
  
  - You can do that because a digital signature is created with a private key and only the owner of that key has access to it. Therefore they must have created the signature.
  
  - Integrity as we know, means we want to make sure data hasn't been tampered with by unauthorized parties. 
  
  - And non-repudiation again means that we want to make sure that whoever signed this can't refute the fact that they sent it and signed it because they had to have. Only they have access to the private key.

- Do not provide data confidentiality. Again, that's what encryption does.

- Used with SSL and TLS, applications, scripts, device drivers

- Encrypts a hash value using a private key

- The signature is verified with the matching public key

Example:

 Imagine that we are sending an email message to a colleague and what we want to do is digitally sign it so that the recipient, our colleague, can rest assured the message really came from us?

1. So what happens is our email software would generate a hash value of the message content by using a hashing algorithm. 

2. Then, we encrypt that hash value using our private key. Not the recipient's, using ours.

3. And so this signature is verified on the other end with the mathematically related public key. That would be our public key that the recipient would have.

from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord

Entrez.email = "ahdezneytiry@gmail.com"

def main():
    q = "antimicrobial peptide"
    handle = Entrez.esearch(db="nucleotide", term=q, retmax=3)
    result = Entrez.read(handle)
    handle.close()

    gene_ids = result.get("IdList", [])
    print("找到 Gene IDs:", gene_ids)
    if not gene_ids:
        print("未找到结果")
        return

    handle = Entrez.efetch(db="nucleotide", id=",".join(gene_ids), rettype="fasta", retmode="text")
    fasta_text = handle.read()
    handle.close()

    with open("AP_gene.fasta", "w", encoding="utf-8") as f:
        f.write(fasta_text)
    print("保存 AP_gene.fasta")

    gene_records = list(SeqIO.parse("AP_gene.fasta", "fasta"))
    protein_records = []
    for rec in gene_records:
        protein_seq = rec.seq.translate(to_stop=False)
        protein_records.append(SeqRecord(
            protein_seq,
            id=rec.id + "_protein",
            description="translated from " + rec.description
        ))

    SeqIO.write(protein_records, "AP_protein.fasta", "fasta")
    print("保存 AP_protein.fasta")

if __name__ == "__main__":
    main()
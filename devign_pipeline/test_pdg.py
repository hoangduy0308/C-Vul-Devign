"""Test PDG and PDG Slicer"""
import sys
sys.path.insert(0, '.')

from src.graphs.pdg import build_pdg, PDG
from src.ast.parser import CFamilyParser
from src.slicing.pdg_slicer import PDGSlicer, PDGSliceConfig, PDGSliceType

# Test with SCSI vulnerability code
test_code = '''static void scsi_read_data(SCSIDiskReq *r) {
    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);
    uint32_t n;
    
    if (r->sector_count == (uint32_t)-1) {
        r->sector_count = 0;
        scsi_req_data(&r->req, r->iov.iov_len);
        return;
    }
    
    if (r->sector_count == 0) {
        scsi_command_complete(r, GOOD, NO_SENSE);
    }
    
    assert(r->req.aiocb == NULL);
    n = r->sector_count;
    if (n > SCSI_DMA_BUF_SIZE / 512)
        n = SCSI_DMA_BUF_SIZE / 512;
    
    r->iov.iov_len = n * 512;
    r->req.aiocb = bdrv_aio_readv(s->bs, r->sector, &r->qiov, n, scsi_read_complete, r);
}'''

print("=== Test PDG Builder ===")
parser = CFamilyParser()
result = parser.parse_with_fallback(test_code)
print(f"Parse result nodes: {len(result.nodes) if result else 0}")

if result:
    pdg = build_pdg(result)
    if pdg:
        print(f"PDG nodes: {len(pdg.nodes)}")
        print(f"PDG edges: {len(pdg.edges)}")
        
        print("\nData edges:")
        from src.graphs.pdg import PDGEdgeType
        for edge in pdg.edges:
            if edge.edge_type == PDGEdgeType.DATA_DEP:
                print(f"  L{edge.from_id} -> L{edge.to_id} ({edge.label})")
        
        print("\nControl edges:")
        for edge in pdg.edges:
            if edge.edge_type == PDGEdgeType.CONTROL_DEP:
                print(f"  L{edge.from_id} -> L{edge.to_id}")
    else:
        print("PDG is None - testing slicer fallback")

# Test PDG Slicer
print("\n=== PDG Slicer Test ===")
print(f"Original code: {len(test_code.splitlines())} lines")

criterion = [12]  # scsi_command_complete line

config = PDGSliceConfig(
    slice_type=PDGSliceType.BIDIRECTIONAL,
    backward_depth=2,
    forward_depth=1,
    control_predicate_only=True,
    max_lines=10,
    fallback_window=3
)

slicer = PDGSlicer(config)
slice_result = slicer.slice(test_code, criterion)

print(f"\nCriterion: line {criterion}")
print(f"Slice lines: {sorted(slice_result.included_lines)}")
print(f"Slice size: {len(slice_result.included_lines)} lines")
print(f"Data deps: {sorted(slice_result.data_dep_lines)}")
print(f"Control deps: {sorted(slice_result.control_dep_lines)}")
print(f"Used fallback: {slice_result.used_fallback}")
print(f"\nSliced code:\n{slice_result.code}")

# Compare with window-based
print("\n=== Comparison ===")
from src.slicing.slicer import CodeSlicer, SliceConfig, SliceType
old_config = SliceConfig(slice_type=SliceType.BACKWARD, window_size=15)
old_slicer = CodeSlicer(old_config)
old_result = old_slicer.slice(test_code, criterion)

print(f"PDG slice: {len(slice_result.included_lines)} lines")
print(f"Window-15 slice: {len(old_result.included_lines)} lines")
print(f"Reduction: {100 - (len(slice_result.included_lines) / max(1, len(old_result.included_lines)) * 100):.0f}%")

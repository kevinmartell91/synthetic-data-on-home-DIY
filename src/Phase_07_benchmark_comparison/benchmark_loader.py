"""Load and prepare benchmark dataset from HuggingFace"""

import random
from typing import List, Dict, Any
from ..pydantic_classes import DIYRepairSyntheticItem, Metadata


class BenchmarkLoader:
    """
    Loads the HuggingFace benchmark dataset: dipenbhuva/home-diy-repair-qa
    Converts to DIYRepairSyntheticItem format for evaluation
    """

    BENCHMARK_DATASET = "dipenbhuva/home-diy-repair-qa"
    REPAIR_CATEGORIES = [
        "appliance_repair",
        "plumbing_repair",
        "electrical_repair",
        "hvac_repair",
        "general_home_repair",
    ]

    @staticmethod
    def load_benchmark_dataset(num_samples: int = 50) -> List[DIYRepairSyntheticItem]:
        """
        Load benchmark dataset from HuggingFace and convert to DIYRepairSyntheticItem

        Args:
            num_samples: Number of samples to load (default 50, minimum for calibration)

        Returns:
            List of DIYRepairSyntheticItem objects from benchmark
        """
        print("HEtet")
        try:
            from datasets import load_dataset
        except ImportError:
            print("⚠️  'datasets' library not found. Install with: pip install datasets")
            # Return mock data for testing
            return BenchmarkLoader._create_mock_benchmark(num_samples)

        try:
            dataset = load_dataset(BenchmarkLoader.BENCHMARK_DATASET, split="train")
        except Exception as e:
            print(
                f"⚠️  Could not load HuggingFace dataset '{BenchmarkLoader.BENCHMARK_DATASET}': {e}"
            )
            print("   Using mock benchmark data instead")
            return BenchmarkLoader._create_mock_benchmark(num_samples)

        # Sample items (stratified by category if possible)
        samples = []
        dataset_list = list(dataset)
        print("dataset", dataset)

        # Try to get balanced samples across categories
        samples_per_category = max(
            1, num_samples // len(BenchmarkLoader.REPAIR_CATEGORIES)
        )

        for category in BenchmarkLoader.REPAIR_CATEGORIES:
            category_items = [
                item
                for item in dataset_list
                if item.get("category", "").lower() == category.lower()
            ]
            selected = random.sample(
                category_items, min(samples_per_category, len(category_items))
            )
            samples.extend(selected)

        # Fill remaining slots randomly if needed
        if len(samples) < num_samples:
            remaining_needed = num_samples - len(samples)
            unused = [item for item in dataset_list if item not in samples]
            samples.extend(random.sample(unused, min(remaining_needed, len(unused))))

        # Convert to DIYRepairSyntheticItem
        items = []
        for idx, sample in enumerate(samples[:num_samples]):
            item = BenchmarkLoader._convert_to_item(sample, idx)
            if item:
                items.append(item)

        print(f"✅ Loaded {len(items)} benchmark samples")
        return items

    @staticmethod
    def _convert_to_item(sample: Dict[str, Any], idx: int) -> DIYRepairSyntheticItem:
        """Convert HuggingFace item to DIYRepairSyntheticItem"""
        try:
            # Extract fields from benchmark item
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            category = sample.get("category", "general_home_repair").lower()

            # Map category to standard format
            if "appliance" in category:
                issue_type = "appliance_repair"
            elif "plumb" in category:
                issue_type = "plumbing_repair"
            elif "electric" in category:
                issue_type = "electrical_repair"
            elif "hvac" in category or "air" in category:
                issue_type = "hvac_repair"
            else:
                issue_type = "general_home_repair"

            # Parse answer into components (best effort)
            # Benchmark data may not have structured fields
            equipment_problem = sample.get("equipment_problem", question[:100])
            tools_required = sample.get("tools_required", ["generic tool"])
            if isinstance(tools_required, str):
                tools_required = [
                    t.strip() for t in tools_required.split(",") if t.strip()
                ]
            if not tools_required:
                tools_required = ["generic tool"]

            steps = sample.get("steps", [answer])
            if isinstance(steps, str):
                steps = [s.strip() for s in steps.split(".") if s.strip()][:5]
            if not steps:
                steps = [answer]

            safety_info = sample.get("safety_info", "Follow standard safety practices")
            tips = sample.get("tips", "Consult an expert if unsure")

            # Create item
            item = DIYRepairSyntheticItem(
                id=f"benchmark_{idx:04d}",
                question=question,
                answer=answer,
                equipment_problem=equipment_problem,
                tools_required=tools_required,
                steps=steps,
                safety_info=safety_info,
                tips=tips,
                metadata=Metadata(
                    issue_type=issue_type,
                    user_query=f"Benchmark: {category}",
                ),
                error=None,
            )
            return item
        except Exception as e:
            print(f"⚠️  Could not convert benchmark item {idx}: {e}")
            return None

    @staticmethod
    def _create_mock_benchmark(num_samples: int) -> List[DIYRepairSyntheticItem]:
        """Create mock benchmark data for testing when real data unavailable"""
        mock_items = [
            {
                "id": f"benchmark_{i:04d}",
                "question": f"How do I fix a {category.replace('_repair', '')} issue?",
                "answer": f"To fix a {category.replace('_repair', '')} issue, follow these steps carefully. "
                f"First, ensure safety by turning off power. Use appropriate tools and materials. "
                f"Follow the repair process step by step.",
                "equipment_problem": f"The {category.replace('_repair', '')} is malfunctioning",
                "tools_required": ["wrench", "screwdriver", "safety glasses"],
                "steps": [
                    "Turn off power and disconnect",
                    "Inspect the component",
                    "Replace or repair as needed",
                    "Test the fix",
                    "Turn back on carefully",
                ],
                "safety_info": "Always turn off power before working on electrical or mechanical components. "
                "Wear safety glasses and gloves.",
                "tips": "Take photos before disassembly. Check warranty before attempting repair.",
                "issue_type": category,
            }
            for i, category in enumerate(
                [
                    "appliance_repair",
                    "plumbing_repair",
                    "electrical_repair",
                    "hvac_repair",
                ]
                * (num_samples // 4 + 1)
            )[:num_samples]
        ]

        items = []
        for data in mock_items:
            item = DIYRepairSyntheticItem(
                id=data["id"],
                question=data["question"],
                answer=data["answer"],
                equipment_problem=data["equipment_problem"],
                tools_required=data["tools_required"],
                steps=data["steps"],
                safety_info=data["safety_info"],
                tips=data["tips"],
                metadata=Metadata(
                    issue_type=data["issue_type"],
                    user_query=f"Benchmark: {data['issue_type']}",
                ),
                error=None,
            )
            items.append(item)

        print(f"⚠️  Using mock benchmark data ({len(items)} items)")
        return items
